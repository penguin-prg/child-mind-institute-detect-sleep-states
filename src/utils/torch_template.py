import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2)
        self.ln = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = nn.SiLU()(x)
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


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[: x.size(0), :]
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = [model(x) for model in self.models]
        preds = torch.stack(preds, dim=0)
        preds = torch.mean(preds, dim=0)
        return preds
