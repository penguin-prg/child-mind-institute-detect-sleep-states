import numpy as np
import pytorch_lightning as pl
import torch


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
