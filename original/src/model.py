import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gMLP import gMLP

acv = nn.GELU()


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
    gt_pw_dif = torch.sub(all_one @ ground_truth.t(), ground_truth @ all_one.t())
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, ((d_model + 1) // 2) * 2).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :-1]


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TimeMixing(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid()
        )
        self.acv = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        gate = self.gate(x)
        x = self.acv(self.mlp1(x))

        x = self.mlp2(x * gate)
        return x


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([
            nn.Linear(i + 1, 1)
            for i in range(time_step)])

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0: i + 1])], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    def __init__(self, time_step):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module):
    def __init__(self, time_step, channel, scale_count=1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channel,
                        out_channels=channel,
                        kernel_size=2 ** i,
                        stride=2 ** i,
                    ),
                    TriU(int(time_step / 2 ** i)),
                    nn.Hardswish(),
                    TriU(int(time_step / 2 ** i)),
                )
                for i in range(scale_count)
            ]
        )
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)
        x = self.LN_2(x + inputs)

        y = self.channelMixer(x)
        return x + y


class LagScale(nn.Module):
    def __init__(self, timestep, channel, scale):
        super(LagScale, self).__init__()
        self.timestep = timestep
        self.scale = scale
        self.conv = nn.Conv2d(channel, channel, kernel_size=(2, 1))
        self.acv = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], self.scale, self.timestep // self.scale)
        # padding = torch.zeros([x.shape[0], x.shape[1], x.shape[2], 1]).to(x.device)
        # x = torch.cat([x, padding], dim=3)
        x = self.conv(x)

        x = x.squeeze(dim=2)
        x = self.acv(x)
        x = x.permute(0, 2, 1)
        return x


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale1_mix_layer = Mixer2dTriU(time_step // 2, channel)

    def forward(self, inputs, x1):
        x = self.mix_layer(inputs)
        x1 = self.scale1_mix_layer(x1)
        return torch.cat([inputs, x, x1], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, time_step, stocks, hidden_dim=20, depth=2):
        super(NoGraphMixer, self).__init__()
        self.time_step = time_step
        # self.dense1 = nn.Linear(stocks, hidden_dim)
        # self.activation = nn.Hardswish()
        # self.dense2 = nn.Linear(hidden_dim , stocks)
        # self.layer_norm_stock = nn.LayerNorm(stocks)
        #
        # self.dense3 = nn.Linear(stocks, hidden_dim)
        # self.gate = nn.Sigmoid()

        self.gMlp = gMLP(time_step * 2 + time_step // 2, stocks, hidden_dim, depth)

    def forward(self, inputs):
        # [x0, x1, x2] = torch.split(inputs, [self.time_step, self.time_step, self.time_step // 2], dim=-1)

        # x = inputs
        # x = x.permute(1, 0)
        # x = self.layer_norm_stock(x)
        #
        # y = self.dense3(x)
        # y = self.gate(y)
        #
        # x = self.dense1(x)
        # x = self.activation(x)
        # x = x * y
        #
        # x = self.dense2(x)
        # x = x.permute(1, 0)

        x = inputs
        x = x.permute(1, 0)
        # x = x.unsqueeze(0)

        x = self.gMlp(x)

        x = x.permute(1, 0)

        return x


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, depth=2):
        super(StockMixer, self).__init__()
        # self.ln = nn.LayerNorm(channels)
        self.mixer = MultTime2dMixer(time_steps, channels)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)
        self.scale1 = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(time_steps, stocks, market, depth)
        self.time_fc_ = nn.Linear(time_steps * 2 + time_steps // 2, 1)

    def forward(self, inputs):
        # inputs = self.ln(inputs)
        # Test
        x1 = inputs.permute(0, 2, 1)
        x1 = self.scale1(x1)
        x1 = x1.permute(0, 2, 1)

        y = self.mixer(inputs, x1)
        y = self.channel_fc(y).squeeze(-1)

        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return z + y
