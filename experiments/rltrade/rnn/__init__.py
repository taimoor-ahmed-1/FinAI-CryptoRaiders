"""
Sequence-model registry. Every architecture is called the same way:
    out, hid = net(x)      x: (batch, time, features) -> out: (batch, time, outputs)

LOOK-AHEAD LEAK FIX - `enhanced`
--------------------------------
The notebook's default model, `EnhancedRnnRegNet`, lets the future reach the past
through TWO separate paths:

  1. Multi-head attention is applied with no mask (`self.attention(x)`), so every
     output step attends over the WHOLE window, future bars included.
  2. The stacked RNN blocks hand state forward in depth:
         x, hidden_states = block(x, hidden_states)
     Block i's FINAL hidden state - which has already read the entire window -
     becomes block i+1's INITIAL state. Block i+1's output at step 0 therefore
     depends on the last input. This path bypasses attention entirely.

Consequences: in training the model can "predict" future price moves by reading
future inputs, so its loss falls without it learning anything tradable; and the
8 factors it produces for the trading agent at time t contain information from
after t, so the agent's state sees the future and every backtest built on it is
inflated.

`tests/test_causality.py` proves both: rewriting only future inputs moves the
notebook model's earlier outputs; masking attention alone shrinks that ~50x but
does not remove it; fixing both paths removes it exactly.

`enhanced` is now `CausalEnhancedRnnRegNet`: identical layers and weights, both
paths closed. The leaky original stays available as `enhanced_legacy`, only for
reproducing earlier results - never use it for new experiments.
"""
import torch as th
import torch.nn as nn

from .models import EnhancedRnnRegNet, RnnRegNet


class CausalEnhancedRnnRegNet(EnhancedRnnRegNet):
    """EnhancedRnnRegNet with both look-ahead paths closed. Same layers, same weights.

    1. Attention gets a causal (lower-triangular) mask.
    2. Each RNN block starts from a fresh hidden state instead of inheriting the
       previous block's FINAL state (which had already read the whole window).
    """

    def forward(self, inp, hid=None):
        _, seq_len, _ = inp.shape
        x = self.input_projection(inp)
        states = []
        for i, block in enumerate(self.rnn_blocks):
            x, st = block(x, None)          # fix 2: was block(x, <previous block's final state>)
            states.append(st)
            if i > 0:
                x = x + residual  # noqa: F821  (mirrors the notebook's residual logic)
            residual = x
        causal = th.tril(th.ones(seq_len, seq_len, device=inp.device))   # fix 1: 1 = may attend
        x = self.attention(x, mask=causal)
        return self.output_mlp(x), states


class _SeqFirst(nn.Module):
    """Adapter: lets the notebook's sequence-first RnnRegNet take batch-first input."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, hid=None):
        out, hid = self.net(x.transpose(0, 1), hid)
        return out.transpose(0, 1), hid


# Defaults: `enhanced` uses the notebook's exact configuration.
DEFAULTS = {
    "enhanced":        dict(mid_dim=192, num_layers=4, num_blocks=2, num_heads=6, dropout=0.35),
    "enhanced_legacy": dict(mid_dim=192, num_layers=4, num_blocks=2, num_heads=6, dropout=0.35),
    "rnnreg":          dict(mid_dim=192, num_layers=4),
    "lstm":            dict(mid_dim=128, num_layers=2, dropout=0.2),
    "gru":             dict(mid_dim=128, num_layers=2, dropout=0.2),
    "transformer":     dict(mid_dim=128, num_layers=2, dropout=0.1, num_heads=4),
    "tcn":             dict(mid_dim=128, num_layers=5, dropout=0.1, kernel=3),
}
ARCHITECTURES = tuple(DEFAULTS)
CAUSAL = {a: a != "enhanced_legacy" for a in ARCHITECTURES}


def build_rnn(arch, inp_dim, out_dim, **hp):
    """Instantiate an architecture by name; `hp` overrides DEFAULTS[arch]."""
    if arch not in DEFAULTS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHITECTURES}")
    cfg = {**DEFAULTS[arch], **hp}
    if arch.startswith("enhanced") and cfg["mid_dim"] % cfg["num_heads"]:
        raise ValueError(f"mid_dim={cfg['mid_dim']} must be divisible by num_heads={cfg['num_heads']}")
    if arch == "enhanced":
        return CausalEnhancedRnnRegNet(inp_dim, out_dim=out_dim, **cfg)
    if arch == "enhanced_legacy":
        return EnhancedRnnRegNet(inp_dim, out_dim=out_dim, **cfg)
    if arch == "rnnreg":
        return _SeqFirst(RnnRegNet(inp_dim, cfg["mid_dim"], out_dim, cfg["num_layers"]))
    from . import zoo
    cls = {"lstm": zoo.LSTMNet, "gru": zoo.GRUNet,
           "transformer": zoo.TransformerNet, "tcn": zoo.TCNNet}[arch]
    return cls(inp_dim, out_dim=out_dim, **cfg)
