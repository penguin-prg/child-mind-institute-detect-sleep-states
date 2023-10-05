from torch import nn
import torch
import torch.nn.functional as F


class Conv1dBnRelu(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Conv1dBnRelu, self).__init__()
        self.conv3 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=5, padding=2, stride=1)
        self.conv7 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm1d(out_chans)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = torch.cat(
            [
                self.conv3(x),
                self.conv5(x),
                self.conv7(x),
            ],
            dim=1,
        )
        x = self.bn(x)
        x = self.relu(x)
        return x


class WaveBlock(nn.Module):
    """from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver"""

    def __init__(self, in_channels, out_channels, dilation_rates):
        super(WaveBlock, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
            )
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x)) * F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            # x += res
            res = torch.add(res, x)
        return res
