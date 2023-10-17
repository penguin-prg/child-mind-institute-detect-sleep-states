import math
from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import MetricCollection


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


class ZzzWaveGRUModule(pl.LightningModule):
    def __init__(
        self,
        dropout=0.2,
        input_numerical_size=2,
        numeraical_linear_size=64,
        model_size=128,
        linear_out=128,
        out_size=2,
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.001,
        weight_decay=0,
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
        self._reinitialize()

        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics = MetricCollection([], prefix="")
        self.valid_metrics = MetricCollection([], prefix="val_")

        self.val_step_outputs = []
        self.val_step_labels = []

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "rnn" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

    def forward(self, feat):
        x = self.numerical_linear(feat)

        x = x.permute(0, 2, 1)
        x = self.wavenet(x)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.linear_out(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        self.train_metrics(preds, y)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        self.val_step_outputs.append(preds)
        self.val_step_labels.append(y)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_step_outputs)
        labels = torch.cat(self.val_step_labels)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        loss = self.loss_fn(preds, labels)

        self.valid_metrics(preds, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log_dict(
            self.valid_metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        # ログをprint
        self.print_metric(preds, labels, "valid")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def print_metric(self, y_hat, y, train_or_valid="train"):
        """
        ログをprintする。次のepochが終わると上書きされてしまうので。
        TODO: たぶんもっとマシな方法があるので探す。
        """
        if train_or_valid == "train":
            metrics = self.train_metrics
        else:
            metrics = self.valid_metrics
        loss = self.loss_fn(y_hat, y)

        print(f"[epoch {self.trainer.current_epoch}] {train_or_valid}: ", end="")
        print(f"{type(self.loss_fn).__name__}={loss:.4f}", end=", ")
        for name in metrics:
            v = metrics[name](y_hat, y)
            print(f"{name}={v:.4f}", end=", ")
        print()


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


class ZzzTransformerGRUModule(pl.LightningModule):
    def __init__(
        self,
        max_len,
        dropout=0.2,
        input_numerical_size=2,
        numeraical_linear_size=84,
        model_size=128,
        linear_out=128,
        out_size=2,
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.001,
        weight_decay=0,
    ):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            nn.Linear(input_numerical_size, numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size),
        )

        self.pe = PositionalEncoding(numeraical_linear_size, dropout=0.0, max_len=max_len, batch_first=True)
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=numeraical_linear_size, nhead=6, dropout=0.0, dim_feedforward=128, batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=numeraical_linear_size, nhead=6, dropout=0.0, dim_feedforward=128, batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=numeraical_linear_size, nhead=6, dropout=0.0, dim_feedforward=128, batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=numeraical_linear_size, nhead=6, dropout=0.0, dim_feedforward=128, batch_first=True
            ),
        )

        self.rnn = nn.GRU(numeraical_linear_size, model_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_out = nn.Sequential(
            nn.Linear(model_size * 2, linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_out, out_size),
        )
        self._reinitialize()

        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics = MetricCollection([], prefix="")
        self.valid_metrics = MetricCollection([], prefix="val_")

        self.val_step_outputs = []
        self.val_step_labels = []

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "rnn" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

    def forward(self, x):
        x = self.numerical_linear(x)

        x = self.pe(x)
        x = self.transformer(x)

        x, _ = self.rnn(x)
        x = self.linear_out(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        loss = self.loss_fn(preds, y)

        self.train_metrics(preds, y)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)

        self.val_step_outputs.append(preds)
        self.val_step_labels.append(y)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_step_outputs)
        labels = torch.cat(self.val_step_labels)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        loss = self.loss_fn(preds, labels)

        self.valid_metrics(preds, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log_dict(
            self.valid_metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        # ログをprint
        self.print_metric(preds, labels, "valid")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def print_metric(self, y_hat, y, train_or_valid="train"):
        """
        ログをprintする。次のepochが終わると上書きされてしまうので。
        TODO: たぶんもっとマシな方法があるので探す。
        """
        if train_or_valid == "train":
            metrics = self.train_metrics
        else:
            metrics = self.valid_metrics
        loss = self.loss_fn(y_hat, y)

        print(f"[epoch {self.trainer.current_epoch}] {train_or_valid}: ", end="")
        print(f"{type(self.loss_fn).__name__}={loss:.4f}", end=", ")
        for name in metrics:
            v = metrics[name](y_hat, y)
            print(f"{name}={v:.4f}", end=", ")
        print()
