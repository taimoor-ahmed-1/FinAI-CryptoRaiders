"""Network building blocks shared by all agents."""
import torch.nn as nn


def mlp(inp_dim, hidden, out_dim, layer_norm=True, dropout=0.0, out_gain=1.0):
    """Linear -> ReLU (-> Dropout) (-> LayerNorm) per hidden layer, then a linear head.

    Mirrors the notebook's actor/critic layout (LayerNorm after every hidden layer
    except the last). Hidden layers use orthogonal init with gain sqrt(2), the head
    uses `out_gain` (0.01 for a policy head keeps the initial policy near uniform).
    The notebook initialised EVERY actor layer with gain 0.01, which shrinks
    activations layer by layer; only the head should be small.
    """
    layers, d = [], inp_dim
    for i, h in enumerate(hidden):
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if layer_norm and i < len(hidden) - 1:
            layers.append(nn.LayerNorm(h))
        d = h
    head = nn.Linear(d, out_dim)
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)
