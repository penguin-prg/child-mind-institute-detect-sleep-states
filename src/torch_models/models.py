import math
import os
import sys
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from transformers import get_cosine_schedule_with_warmup

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)

from utils.torch_template import Conv1dBlock, PositionalEncoding, WaveBlock


class ZzzWaveGRUModel(nn.Module):
    def __init__(
        self,
        dropout=0.2,
        input_numerical_size=2,
        numeraical_linear_size=64,
        model_size=128,
        linear_out=128,
        out_size=2,
    ):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            nn.Linear(input_numerical_size, numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size),
        )

        self.wavenet = nn.Sequential(
            WaveBlock(numeraical_linear_size, 16, 4),
            WaveBlock(16, 32, 4),
            WaveBlock(32, 64, 2),
            WaveBlock(64, numeraical_linear_size, 1),
        )

        self.rnn = nn.GRU(numeraical_linear_size, model_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_out = nn.Sequential(
            nn.Linear(model_size * 2, linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_out, out_size),
        )

    def forward(self, feat):
        x = self.numerical_linear(feat)

        x = x.permute(0, 2, 1)
        x = self.wavenet(x)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.linear_out(x)
        return x


class ZzzTransformerGRUModel(nn.Module):
    def __init__(
        self,
        max_len,
        dropout=0.0,
        input_numerical_size=2,
        numeraical_linear_size=84,  # TF
        num_layers=4,  # TF
        dim_feedforward=128,  # TF
        model_size=128,  # GRU
        linear_out=128,  # head
        out_size=2,
    ):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            nn.Linear(input_numerical_size, numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numeraical_linear_size, numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size),
        )

        self.pe = PositionalEncoding(numeraical_linear_size, dropout=0.0, max_len=max_len, batch_first=True)
        self.transformer = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=numeraical_linear_size,
                    nhead=6,
                    dropout=0.0,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.rnn = nn.GRU(numeraical_linear_size, model_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_out = nn.Sequential(
            nn.Linear(model_size * 2, linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_out, out_size),
        )

    def forward(self, x):
        x = self.numerical_linear(x)

        x = self.pe(x)
        x = self.transformer(x)

        x, _ = self.rnn(x)
        x = self.linear_out(x)
        return x


class ZzzConv1dGRUModel(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        input_numerical_size=2,
        numeraical_linear_size=84,
        model_size=128,  # GRU
        linear_out=128,  # head
        out_size=2,
    ):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            nn.Linear(input_numerical_size, numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size),
        )

        k = 64
        assert numeraical_linear_size % 6 == 0
        out_ch = numeraical_linear_size // 8
        self.c1 = Conv1dBlock(numeraical_linear_size, out_ch, k - 1)
        self.c2 = Conv1dBlock(numeraical_linear_size, out_ch, k * 2 - 1)
        self.c3 = Conv1dBlock(numeraical_linear_size, out_ch, k // 2 - 1)
        self.c4 = Conv1dBlock(numeraical_linear_size, out_ch, k // 4 - 1)
        self.c5 = Conv1dBlock(numeraical_linear_size, out_ch, k * 4 - 1)
        self.c6 = Conv1dBlock(numeraical_linear_size, out_ch, k * 8 - 1)
        self.c7 = Conv1dBlock(numeraical_linear_size, out_ch, k * 16 - 1)
        self.c8 = Conv1dBlock(out_ch * 7 + numeraical_linear_size, numeraical_linear_size, 1)

        self.rnn = nn.GRU(numeraical_linear_size, model_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_out = nn.Sequential(
            nn.Linear(model_size * 2, linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_out, out_size),
        )

    def forward(self, x):
        x = self.numerical_linear(x)

        x = x.permute(0, 2, 1)
        x = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), self.c5(x), self.c6(x), self.c7(x), x], dim=1)
        x = self.c8(x)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.linear_out(x)
        return x
