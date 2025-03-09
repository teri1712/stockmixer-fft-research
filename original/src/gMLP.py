import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split channels
        u, v = torch.chunk(x, chunks=2, dim=-1)

        # Apply normalization and spatial projection to v
        v = self.norm(v)
        v = v.transpose(-1, -2)  # [batch, dim, seq_len]

        v = self.spatial_proj(v)  # [batch, dim, seq_len]
        v = self.sigmoid(v)
        v = v.transpose(-1, -2)  # [batch, seq_len, dim]

        # Element-wise multiplication with u
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, dim, seq_len, expansion_factor=4, dropout_rate=0.0):
        super().__init__()

        hidden_dim = dim * expansion_factor
        self.norm = nn.LayerNorm(dim)
        self.channel_proj1 = nn.Linear(dim, hidden_dim * 2)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(hidden_dim, seq_len)
        self.channel_proj2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        # Norm and first projection
        x = self.norm(x)
        x = self.channel_proj1(x)
        x = self.activation(x)

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
            dim=128,
            depth=6,
            expansion_factor=2,
            dropout_rate=0.1
    ):
        super().__init__()

        self.proj_in = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([
            gMLPBlock(
                dim=dim,
                seq_len=seq_len,
                expansion_factor=expansion_factor,
                dropout_rate=dropout_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, input_dim)

    def forward(self, x):
        # Initial projection [batch, seq_len, input_dim] -> [batch, seq_len, dim]
        x = self.proj_in(x)
        # Process through gMLP blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)
        # Classification
        return self.proj_out(x)
