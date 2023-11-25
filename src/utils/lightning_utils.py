import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from transformers import get_cosine_schedule_with_warmup
import gc
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from utils.metric import compute_comptetition_metric
from utils.postprocess import dynamic_range_nms


class MyLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset=None, valid_dataset=None, test_dataset=None, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
            num_workers=0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
            num_workers=0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
            num_workers=0,
        )

    def worker_init_fn(self, worker_id):
        # dataloaderでnum_workers>1の時の乱数設定
        # これを指定しないと各workerのrandom_stateが同じになり、データも同じになる。
        np.random.seed(np.random.get_state()[1][0] + worker_id)


class MyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn=nn.CrossEntropyLoss(),
        lr: float = 0.001,
        num_training_steps: int = 1000,
        weight_decay: float = 0,
    ):
        super().__init__()

        self.model = model
        self._reinitialize()

        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps

        self.train_metrics = MetricCollection([], prefix="")
        self.valid_metrics = MetricCollection([], prefix="val_")

        self.val_step_outputs = []
        self.val_step_labels = []
        self.val_step_keys = []

    def forward(self, x):
        return self.model(x)

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

    def training_step(self, batch, batch_idx):
        X = batch["feats"]
        y = batch["targets"]
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
        X = batch["feats"]
        y = batch["targets"]
        keys = batch["keys"]
        preds = self.forward(X)

        self.val_step_outputs.append(preds)
        self.val_step_labels.append(y)
        self.val_step_keys += keys

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_step_outputs)
        labels = torch.cat(self.val_step_labels)
        keys = self.val_step_keys
        gc.collect()
        loss = self.loss_fn(preds, labels)
        score = self.compute_metric(preds, labels, keys)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        self.val_step_keys.clear()

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
        self.log(
            "val_score",
            score,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        # ログをprint
        self.print_metric(preds, labels, "valid")
        print(f"val_score={score:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, self.num_training_steps)

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
        print(f"{type(self.loss_fn).__name__}={loss:.6f}", end=", ")
        for name in metrics:
            v = metrics[name](y_hat, y)
            print(f"{name}={v:.6f}", end=", ")
        print()

    @classmethod
    def compute_metric(cls, preds, labels, keys) -> float:
        dfs = []
        for key, pred, _ in zip(keys, preds, labels):
            sid = key.split("_")[0]
            start = int(key.split("_")[1])
            end = int(key.split("_")[2])
            df = pd.DataFrame(
                {
                    "series_id": sid,
                    "step": np.arange(start + 6, end, 12),
                    "wakeup_oof": pred[:, 0].sigmoid().detach().cpu().numpy(),
                    "onset_oof": pred[:, 1].sigmoid().detach().cpu().numpy(),
                }
            )
            dfs.append(df)
        df = pd.concat(dfs)
        train = df.groupby(["series_id", "step"]).mean().reset_index().sort_values(["series_id", "step"])

        labels = pd.read_csv("/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv")
        labels = labels[labels["series_id"].isin(train["series_id"].unique())].dropna()
        labels["step"] = labels["step"].astype(int)

        dfs = []
        df = train[["series_id", "step", "wakeup_oof"]].copy()
        df["event"] = "wakeup"
        df["score"] = df["wakeup_oof"]
        dfs.append(df[["series_id", "step", "event", "score"]])

        df = train[["series_id", "step", "onset_oof"]].copy()
        df["event"] = "onset"
        df["score"] = df["onset_oof"]
        dfs.append(df[["series_id", "step", "event", "score"]])

        train = pd.concat(dfs)
        train = train[train["score"] > 0.005].reset_index(drop=True)
        if len(train) == 0:
            return 0.0

        train["step"] = train["step"].astype(int)
        groups = [group for _, group in train.groupby("series_id")]
        with Pool(30) as p:
            results = list(p.imap(dynamic_range_nms, groups))
        sub = pd.concat(results)
        sub["score"] = sub["reduced_score"]
        score = compute_comptetition_metric(labels, sub)[0]
        return score
