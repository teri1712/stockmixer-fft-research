from torch import nn
from torch.nn import functional as F
import torch


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([nn.Linear(i + 1, 1) for i in range(time_step)])

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0 : i + 1])], dim=-1)
        return x


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden, tokens):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.spatial_proj = TriU(tokens)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.spatial_proj(v)
        v = v.permute(0, 2, 1)
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
    def __init__(self, features, tokens, expand=2, num_layers=2):
        super().__init__()
        self.hidden = features * expand
        self.model = nn.Sequential(
            *[gMLPBlock(features, self.hidden, tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)
