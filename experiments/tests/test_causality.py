"""No architecture used for experiments may let the future influence the past.

Method: run a sequence through the model, then change ONLY inputs at steps
>= t_cut and run it again. In a causal model, outputs before t_cut are
unchanged. Any difference there means the model reads future inputs - look-ahead.

The notebook's original `enhanced_legacy` model is checked too, to document that
it is NOT causal (the leak this project fixes).
"""
import sys
from pathlib import Path

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.rnn import ARCHITECTURES, CAUSAL, build_rnn   # noqa: E402
from rltrade.rnn.train import predict_series                # noqa: E402

INP, OUT, T, B, T_CUT = 101, 8, 96, 3, 60


def max_past_change(arch, seed=0):
    th.manual_seed(seed)
    net = build_rnn(arch, INP, OUT).eval()
    x = th.randn(B, T, INP)
    x2 = x.clone()
    x2[:, T_CUT:, :] = th.randn(B, T - T_CUT, INP) * 5      # rewrite the future only
    with th.no_grad():
        y1, _ = net(x)
        y2, _ = net(x2)
    assert y1.shape == (B, T, OUT), f"{arch}: output shape {tuple(y1.shape)}"
    return (y1[:, :T_CUT] - y2[:, :T_CUT]).abs().max().item()


def test_every_experiment_architecture_is_causal():
    for arch in ARCHITECTURES:
        if not CAUSAL[arch]:
            continue
        d = max_past_change(arch)
        assert d < 1e-5, f"{arch} is NOT causal: past outputs moved by {d:.3e}"


def test_factor_prediction_is_causal():
    """The whole factor-writing path (overlapping windows), not just the model."""
    th.manual_seed(0)
    net = build_rnn("enhanced", INP, OUT, mid_dim=48, num_layers=1).eval()
    x = th.randn(1000, INP)
    x2 = x.clone()
    cut = 700
    x2[cut:] = th.randn(1000 - cut, INP) * 5
    y1 = predict_series(net, x, seq_len=128, wup_dim=32)
    y2 = predict_series(net, x2, seq_len=128, wup_dim=32)
    d = abs(y1[:cut] - y2[:cut]).max()
    assert d < 1e-5, f"factor prediction reads the future: {d:.3e}"


def test_attention_mask_is_float16_safe():
    """The causal mask must survive half precision: the GPU trains under float16 autocast.

    The provided attention filled masked scores with -1e9, which does not fit in
    float16 (max ~65504) and crashed the first training step on the T4.
    """
    from rltrade.rnn.models import MultiHeadAttention
    th.manual_seed(0)
    attn = MultiHeadAttention(48, 6, dropout=0.0).half().eval()
    x = th.randn(2, 16, 48).half()
    with th.no_grad():
        y = attn(x, mask=th.tril(th.ones(16, 16)))
    assert th.isfinite(y).all()


def test_notebook_original_leaks():
    d = max_past_change("enhanced_legacy")
    assert d > 1e-4, "expected the notebook's unmasked-attention model to leak"


if __name__ == "__main__":
    print(f"{'architecture':<18}{'max change in past outputs':>28}   verdict")
    for arch in ARCHITECTURES:
        d = max_past_change(arch)
        verdict = "causal" if d < 1e-5 else "LEAKS THE FUTURE"
        print(f"{arch:<18}{d:>28.3e}   {verdict}")
    test_every_experiment_architecture_is_causal()
    test_factor_prediction_is_causal()
    print("factor prediction path (overlapping windows): causal")
    test_attention_mask_is_float16_safe()
    print("causal attention mask: float16-safe")
    test_notebook_original_leaks()
    print("\nOK - every experiment architecture is causal; the notebook original leaks.")
