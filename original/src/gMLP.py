import torch
import torch.nn as nn

import torch.nn.init as init


class SigmoidGatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ln1 = nn.Linear(dim, dim // 2)
        self.ln2 = nn.Linear(dim // 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.acv = nn.Hardswish()

    def forward(self, x):
        # Split channels
        u, v = torch.split(x, [self.dim, self.dim], dim=-1)
        # Apply normalization and spatial projection to v
        # v = self.norm(v)
        # v = v.transpose(-1, -2)  # [batch, dim, seq_len]
        #
        # v = self.spatial_proj(v)  # [batch, dim, seq_len]
        # v = self.ln1(v)
        u = self.acv(u)
        # v = self.ln2(v)
        # u = self.acv(u)
        v = self.sigmoid(v)
        # v = v.transpose(-1, -2)  # [batch, seq_len, dim]

        # Element-wise multiplication with u
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, dropout_rate=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.sgu = SigmoidGatingUnit(hidden_dim, seq_len)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.acv = nn.Hardswish()
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize channel projections with Xavier/Glorot initialization
        init.xavier_uniform_(self.channel_proj1.weight)
        init.xavier_uniform_(self.channel_proj2.weight)
        init.zeros_(self.channel_proj1.bias)
        init.zeros_(self.channel_proj2.bias)

        # Initialize LayerNorm parameters
        init.ones_(self.norm.weight)
        init.zeros_(self.norm.bias)

    def forward(self, x):
        residual = x

        # Norm and first projection
        x = self.norm(x)
        x = self.channel_proj1(x)
        # x = self.acv(x)

        # Apply spatial gating unit
        x = self.sgu(x)

        # Second projection and dropout
        x = self.channel_proj2(x)
        # x = self.dropout(x)

        # Add residual connection
        return x + residual


class gMLP(nn.Module):
    def __init__(
            self,
            seq_len,
            input_dim,
            hidden_dim=128,
            depth=4,
            dropout_rate=0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            gMLPBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                dropout_rate=dropout_rate
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
