"""
Additional sequence architectures for the tuning comparison (brief section 3.3).

All are batch-first and CAUSAL: the output at step t depends only on inputs at
steps <= t. That matters because predictions become state features for a trading
agent - a non-causal model (e.g. a bidirectional RNN or unmasked attention)
would leak the future into the state.

    LSTMNet         plain stacked LSTM
    GRUNet          plain stacked GRU
    TransformerNet  Transformer encoder with a causal attention mask
    TCNNet          temporal convolutional network (dilated causal convolutions)

Output is squashed with tanh, matching the [-1, 1] label range.
"""
import math

import torch as th
import torch.nn as nn


class _RecurrentNet(nn.Module):
    def __init__(self, cell, inp_dim, mid_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.rnn = cell(mid_dim, mid_dim, num_layers=num_layers, batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(mid_dim, mid_dim), nn.GELU(),
                                 nn.Linear(mid_dim, out_dim), nn.Tanh())

    def forward(self, x, hid=None):
        h, hid = self.rnn(self.inp(x), hid)
        return self.out(h), hid


class LSTMNet(_RecurrentNet):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__(nn.LSTM, inp_dim, mid_dim, out_dim, num_layers, dropout)


class GRUNet(_RecurrentNet):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__(nn.GRU, inp_dim, mid_dim, out_dim, num_layers, dropout)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pos = th.arange(max_len).unsqueeze(1)
        div = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, d_model)
        pe[:, 0::2] = th.sin(pos * div)
        pe[:, 1::2] = th.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers=2, dropout=0.1, num_heads=4):
        super().__init__()
        if mid_dim % num_heads:
            num_heads = next(h for h in (8, 6, 4, 3, 2, 1) if mid_dim % h == 0)
        self.inp = nn.Linear(inp_dim, mid_dim)
        self.pos = _PositionalEncoding(mid_dim)
        layer = nn.TransformerEncoderLayer(mid_dim, num_heads, dim_feedforward=mid_dim * 2,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(mid_dim), nn.Linear(mid_dim, out_dim), nn.Tanh())

    def forward(self, x, hid=None):
        t = x.size(1)
        causal = th.triu(th.full((t, t), float("-inf"), device=x.device), diagonal=1)
        h = self.enc(self.pos(self.inp(x)), mask=causal)
        return self.out(h), None


class _CausalConvBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation              # left-pad only -> causal
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, x):                               # x: (B, C, T)
        y = self.drop(self.act(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(self.act(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return self.act(x + y)


class TCNNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers=4, dropout=0.1, kernel=3):
        super().__init__()
        self.inp = nn.Conv1d(inp_dim, mid_dim, 1)
        self.blocks = nn.Sequential(*[_CausalConvBlock(mid_dim, kernel, 2 ** i, dropout)
                                      for i in range(num_layers)])
        self.out = nn.Sequential(nn.Linear(mid_dim, out_dim), nn.Tanh())

    def forward(self, x, hid=None):
        h = self.blocks(self.inp(x.transpose(1, 2))).transpose(1, 2)
        return self.out(h), None

    @property
    def receptive_field(self):
        k = self.blocks[0].conv1.kernel_size[0]
        return 1 + 2 * (k - 1) * sum(2 ** i for i in range(len(self.blocks)))
