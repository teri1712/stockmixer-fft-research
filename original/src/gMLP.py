import torch
import torch.nn as nn


class SigmoidGatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.ln1 = nn.Linear(dim, dim)
        self.ln2 = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.acv = nn.ReLU()

    def forward(self, x):
        # Split channels
        u, v = torch.chunk(x, chunks=2, dim=-1)
        # u = self.acv(u)
        v = self.norm(v)
        v = self.ln1(v)
        v = self.acv(v)
        v = self.ln2(v)
        v = self.sigmoid(v)
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, dropout_rate=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.sgu = SigmoidGatingUnit(hidden_dim, seq_len)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        # self.dropout = nn.Dropout(dropout_rate)
        self.acv = nn.Hardswish()

    def forward(self, x):
        residual = x

        # Norm and first projection
        x = self.norm(x)
        x = self.channel_proj1(x)
        x = self.acv(x)

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
