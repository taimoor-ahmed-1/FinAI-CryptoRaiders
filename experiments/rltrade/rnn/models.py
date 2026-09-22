"""
The provided notebook's recurrent architectures, copied VERBATIM from
`4_rnn_trainer.ipynb`. Nothing in these classes has been edited.

  EnhancedRnnRegNet  - the notebook's default ("current"): stacked LSTM+GRU blocks
                       with residuals and multi-head attention. Batch-first.
  RnnRegNet          - the notebook's simpler LSTM+GRU model. Sequence-first.

`rltrade.rnn.build_rnn` wraps both so every architecture is called the same way
- (batch, time, features) in, (batch, time, outputs) out.
"""
import torch as th
import torch.nn as nn


class NnSeqBnMLP(nn.Module):
    def __init__(self, dims, if_inp_norm=False, if_layer_norm=True, activation=None, dropout=0.1):
        super(NnSeqBnMLP, self).__init__()

        mlp_list = []
        if if_inp_norm:
            mlp_list.append(nn.BatchNorm1d(dims[0], momentum=0.9))

        mlp_list.append(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims) - 1):
            mlp_list.append(nn.GELU())
            mlp_list.append(nn.Dropout(dropout))
            mlp_list.append(nn.LayerNorm(dims[i])) if if_layer_norm else None
            mlp_list.append(nn.Linear(dims[i], dims[i + 1]))

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, seq):
        d0, d1, d2 = seq.shape
        inp = seq.reshape(d0 * d1, -1)
        out = self.mlp(inp)
        return out.reshape(d0, d1, -1)

    def reset_parameters(self, std=1.0, bias_const=1e-6):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                th.nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_const)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, std)
                nn.init.constant_(module.bias, 0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear transformations and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = th.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = th.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = th.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection and residual connection
        output = self.w_o(context)
        output = self.layer_norm(output + x)

        return output

class DeepRNNBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection layer to combine LSTM and GRU outputs
        self.combine_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hidden_states=None):
        lstm_hidden, gru_hidden = (None, None) if hidden_states is None else hidden_states

        # LSTM branch
        lstm_out, lstm_hidden = self.lstm(x, lstm_hidden)
        lstm_out = self.layer_norm1(lstm_out + x)  # Residual connection

        # GRU branch
        gru_out, gru_hidden = self.gru(x, gru_hidden)
        gru_out = self.layer_norm2(gru_out + x)  # Residual connection

        # Combine outputs and project back to original dimension
        combined = th.cat([lstm_out, gru_out], dim=-1)
        combined = self.combine_projection(combined)

        return combined, (lstm_hidden, gru_hidden)

class EnhancedRnnRegNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers, num_blocks=3, num_heads=8, dropout=0.1):
        super(EnhancedRnnRegNet, self).__init__()

        self.num_blocks = num_blocks
        self.mid_dim = mid_dim

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(inp_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Deep RNN blocks with residual connections
        self.rnn_blocks = nn.ModuleList([
            DeepRNNBlock(mid_dim, num_layers, dropout) for _ in range(num_blocks)
        ])

        # Attention mechanism (now works with mid_dim since we project back)
        self.attention = MultiHeadAttention(mid_dim, num_heads, dropout)

        # Output processing
        self.output_mlp = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LayerNorm(mid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim // 2, out_dim),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            th.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, inp, hid=None):
        batch_size, seq_len, _ = inp.shape

        # Input projection
        x = self.input_projection(inp)

        # Process through RNN blocks with residual connections
        hidden_states = hid
        for i, block in enumerate(self.rnn_blocks):
            x, hidden_states = block(x, hidden_states)

            # Add residual connection between blocks
            if i > 0:
                x = x + residual
            residual = x

        # Apply attention
        x = self.attention(x)

        # Output processing
        out = self.output_mlp(x)

        return out, hidden_states

    @staticmethod
    def get_obj_value(criterion, out: th.Tensor, lab: th.Tensor, wup_dim: int) -> th.Tensor:
        obj = criterion(out, lab)[wup_dim:, :, :]
        return obj

# Keep the original RnnRegNet for backward compatibility
class RnnRegNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super(RnnRegNet, self).__init__()
        self.rnn1 = nn.LSTM(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = NnSeqBnMLP(
            dims=(mid_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )

        self.rnn2 = nn.GRU(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp2 = NnSeqBnMLP(
            dims=(mid_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )

        self.mlp_inp = NnSeqBnMLP(
            dims=(inp_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )
        self.mlp_out = NnSeqBnMLP(
            dims=(mid_dim * 2, mid_dim * 2, out_dim),
            if_layer_norm=False,
            activation=nn.Tanh(),
        )

    def forward(self, inp, hid=None):
        hid1, hid2 = (None, None) if hid is None else hid
        inp = self.mlp_inp(inp)

        rnn1, hid1 = self.rnn1(inp, hid1)
        tmp1 = self.mlp1(rnn1)

        rnn2, hid2 = self.rnn2(inp, hid2)
        tmp2 = self.mlp2(rnn2)

        tmp = th.concat((tmp1, tmp2), dim=2)
        out = self.mlp_out(tmp)
        return out, (hid1, hid2)

    @staticmethod
    def get_obj_value(criterion, out: th.Tensor, lab: th.Tensor, wup_dim: int) -> th.Tensor:
        obj = criterion(out, lab)[wup_dim:, :, :]
        return obj

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
