import time
from IPython import embed
from matplotlib.pyplot import sca
from sympy import primefactors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

acv = nn.GELU()


def FFT_for_Period(x, k):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


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
                        kernel_size=2**i,
                        stride=2**i,
                    ),
                    TriU(int(time_step / 2**i)),
                    nn.Hardswish(),
                    TriU(int(time_step / 2**i)),
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


class LagMixer(nn.Module):
    def __init__(self, time_step, channel, embed_dim):
        super(LagMixer, self).__init__()

        self.time_step = time_step
        self.unfold = nn.Unfold(kernel_size=(2, 2), stride=1)

        self.dense_patch_embed = nn.Linear(channel * 4, embed_dim)
        self.acv = nn.GELU()

        # self.flat1 = nn.Flatten(start_dim=1, end_dim=2)
        self.mix_layer = Mixer2dTriU(time_step, embed_dim)

    def forward(self, inputs):
        paddings1 = torch.zeros(
            [inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]
        ).to(inputs.device)
        inputs = torch.cat([inputs, paddings1], dim=1)

        paddings2 = torch.zeros(
            [inputs.shape[0], inputs.shape[1], 1, inputs.shape[3]]
        ).to(inputs.device)
        inputs = torch.cat([inputs, paddings2], dim=2)

        inputs = inputs.permute(0, 3, 1, 2)

        x = self.unfold(inputs)

        x = x.permute(0, 2, 1)

        x = self.dense_patch_embed(x)
        x = self.acv(x)

        # x = self.flat1(x)

        x = x[:, : self.time_step, :]

        x = self.mix_layer(x)
        return x


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, k, embed_dim):
        super(MultTime2dMixer, self).__init__()
        self.k = k
        self.mix_layers = nn.ParameterList(
            [LagMixer(time_step, channel, embed_dim) for i in range(k)]
        )

    def forward(self, inputs):
        window_length = inputs.shape[1]
        scale_list, scale_weight = FFT_for_Period(inputs, self.k)
        outs = []

        for i in range(self.k):
            scale = scale_list[i]
            x = inputs
            if window_length % scale != 0:
                expand_length = scale * (window_length // 2 + 1)

                padding = torch.zeros(
                    [x.shape[0], expand_length - window_length, x.shape[2]]
                ).to(x.device)
                x = torch.cat([inputs, padding], dim=1)

            x = x.reshape(x.shape[0], x.shape[1] // scale, scale, x.shape[2])
            outs.append(self.mix_layers[i](x))
        return torch.cat(outs, dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x


class StockMixer(nn.Module):

    def __init__(self, stocks, time_steps, channels, market, k=3, embed_dim=20):
        super(StockMixer, self).__init__()
        self.mixer = MultTime2dMixer(time_steps, channels, k, embed_dim)
        self.channel_fc = nn.Linear(embed_dim, 1)
        self.time_fc = nn.Linear(time_steps * k, 1)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc_ = nn.Linear(time_steps * k, 1)

    def forward(self, inputs):
        y = self.mixer(inputs)
        y = self.channel_fc(y).squeeze(-1)

        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return y + z
