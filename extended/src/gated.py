from torch import nn
from torch.nn import functional as F


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden, tokens):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.spatial_proj = nn.Conv1d(tokens, tokens, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, features, hidden, tokens):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.channel_proj1 = nn.Linear(features, hidden * 2)
        self.channel_proj2 = nn.Linear(hidden, features)
        self.sgu = SpatialGatingUnit(hidden, tokens)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):
    def __init__(self, features, tokens, expand=1, num_layers=1):
        super().__init__()
        self.hidden = features * expand
        self.model = nn.Sequential(
            *[gMLPBlock(features, expand, tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)
