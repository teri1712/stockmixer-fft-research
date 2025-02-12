import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

        # self.dense3 = nn.Linear(mlp_dim, hidden_dim)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        # y = self.dense3(x)
        # y = self.gate(y)

        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        # x = x * y
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


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

        self.dense3 = nn.Linear(stocks, hidden_dim)
        self.gate = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)

        y = self.dense3(x)
        y = self.gate(y)

        x = self.dense1(x)
        x = self.activation(x)
        x = x * y

        x = self.dense2(x)
        x = x.permute(1, 0)
        return x


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel):
        super(MultTime2dMixer, self).__init__()

        self.scale0_mix_layer = Mixer2dTriU(time_step, channel)
        self.scale1_mix_layer = Mixer2dTriU(time_step // 2, channel)

        # self.gate0 = nn.Sequential(
        #     nn.LayerNorm(time_step),
        #     nn.Linear(time_step, time_step // 2),
        #     nn.ReLU(),
        #     nn.Linear(time_step // 2, 1),
        #     nn.Sigmoid(),
        # )
        # self.gate1 = nn.Sequential(
        #     nn.LayerNorm(time_step // 2),
        #     nn.Linear(time_step // 2, time_step // 4),
        #     nn.ReLU(),
        #     nn.Linear(time_step // 4, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, inputs, x1):

        x0 = inputs
        x0 = self.scale0_mix_layer(x0)
        x1 = self.scale1_mix_layer(x1)

        return torch.cat([inputs, x0, x1], dim=1)


class ScalesFc(nn.Module):
    def __init__(self, time_steps):
        super(ScalesFc, self).__init__()

        self.time_steps = time_steps
        self.fc0 = nn.Linear(time_steps, 1)
        self.fc1 = nn.Linear(time_steps + time_steps, time_steps)
        self.fc2 = nn.Linear(time_steps // 2 + time_steps, time_steps)

    def forward(self, inputs):
        x0, x1, x2 = torch.split(
            inputs, [self.time_steps, self.time_steps, self.time_steps // 2], dim=-1
        )
        x1 = self.fc2(torch.cat([x1, x2], dim=-1))
        x0 = self.fc1(torch.cat([x1, x0], dim=-1))

        return self.fc0(x0)


class BottomUpFc(nn.Module):
    def __init__(self, time_steps):
        super(BottomUpFc, self).__init__()

        self.time_steps = time_steps
        self.fc0 = nn.Sequential(
            nn.LayerNorm(time_steps),
            nn.Linear(time_steps, time_steps),
            nn.ReLU(),
            nn.Linear(time_steps, 1),
        )

        self.fc1 = nn.Sequential(
            nn.LayerNorm(time_steps + time_steps),
            nn.Linear(time_steps + time_steps, time_steps + time_steps),
            nn.ReLU(),
            nn.Linear(time_steps + time_steps, time_steps),
        )

        self.fc2 = nn.Sequential(
            nn.LayerNorm(time_steps + time_steps // 2),
            nn.Linear(
                time_steps + time_steps // 2,
                time_steps + time_steps // 2,
            ),
            nn.ReLU(),
            nn.Linear(time_steps + time_steps // 2, time_steps),
        )

    def forward(self, y, z):

        y0, y1, y2 = torch.split(
            y, [self.time_steps, self.time_steps, self.time_steps // 2], dim=-1
        )
        z0, z1, z2 = torch.split(
            z, [self.time_steps, self.time_steps, self.time_steps // 2], dim=-1
        )
        z1 = self.fc2(torch.cat([z1, z2 + y2], dim=-1))
        z0 = self.fc1(torch.cat([z0, z1 + y1], dim=-1))

        return self.fc0(z0 + y0)


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, scale):
        super(StockMixer, self).__init__()
        self.mixer = MultTime2dMixer(time_steps, channels)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = BottomUpFc(time_steps)

        self.scale1 = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.stock_mixer1 = NoGraphMixer(stocks, market)
        # self.time_fc_ = nn.Linear(time_steps * 2 + time_steps // 2, 1)

    def forward(self, inputs):
        x1 = inputs.permute(0, 2, 1)
        x1 = self.scale1(x1)
        x1 = x1.permute(0, 2, 1)
        y = self.mixer(inputs, x1)
        y = self.channel_fc(y).squeeze(-1)

        z = self.stock_mixer1(y)

        return self.time_fc(y, z)
        # z = self.time_fc_(z)
        # return y + z
