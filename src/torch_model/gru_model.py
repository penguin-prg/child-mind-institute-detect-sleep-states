import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from torchmetrics import MetricCollection


class ZzzGRUModule(pl.LightningModule):
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
            nn.Linear(input_numerical_size, numeraical_linear_size), nn.LayerNorm(numeraical_linear_size)
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
        numerical_embedding = self.numerical_linear(feat)
        output, _ = self.rnn(numerical_embedding)
        output = self.linear_out(output)
        return output

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
