import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import sys
from typing import List
import random


if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)

from utils.feature_contena import Features


class ZzzDataset(Dataset):
    def __init__(self, dfs: List[pd.DataFrame], mode: str, features: Features):
        self.dfs = dfs
        self.mode = mode
        self.features = features

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df = self.dfs[index]

        feats = df[self.features.all_features()].values.astype(np.float32)

        if self.mode == "train":
            targets = df[["wakeup_target", "onset_target"]].values.astype(np.float32)
            return feats, targets
        else:
            return feats


class ZzzPatchDataset(Dataset):
    def __init__(self, dfs: List[pd.DataFrame], mode: str, features: Features, patch_size: int):
        self.dfs = dfs
        self.mode = mode
        self.features = features
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df = self.dfs[index]

        max_len = df.shape[0]
        patch_size = self.patch_size
        n_feats = len(self.features.all_features())

        feats = df[self.features.all_features()].values.astype(np.float32)
        feats = feats.reshape(max_len // patch_size, patch_size, n_feats)
        feats = feats.reshape(max_len // patch_size, patch_size * n_feats)

        if self.mode == "train":
            targets = df[["wakeup_target", "onset_target"]].values.astype(np.float32)
            targets = targets.reshape(max_len // patch_size, patch_size, 2).mean(axis=1)
            assert targets.shape == (max_len // patch_size, 2)
            return feats, targets
        else:
            return feats


class ZzzPatchAugmentationDataset(Dataset):
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        mode: str,
        features: Features,
        patch_size: int,
        aug: bool = False,
        sleeping_dfs: List[pd.DataFrame] = None,
        awake_dfs: List[pd.DataFrame] = None,
    ):
        self.dfs = dfs
        self.mode = mode
        self.features = features
        self.patch_size = patch_size
        self.sleeping_dfs = sleeping_dfs
        self.awake_dfs = awake_dfs
        self.aug = aug

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df = self.dfs[index]

        max_len = df.shape[0]
        patch_size = self.patch_size
        n_feats = len(self.features.all_features())

        # augmentation
        if self.aug and random.random() < 0.3:
            feats = self.copy_and_paste(df)
        else:
            feats = df[self.features.all_features()].values.astype(np.float32)

        feats = feats.reshape(max_len // patch_size, patch_size, n_feats)
        feats = feats.reshape(max_len // patch_size, patch_size * n_feats)

        if self.mode == "train":
            targets = df[["wakeup_target", "onset_target"]].values.astype(np.float32)
            targets = targets.reshape(max_len // patch_size, patch_size, 2).mean(axis=1)
            assert targets.shape == (max_len // patch_size, 2)
            return feats, targets
        else:
            return feats

    def copy_and_paste(self, df: pd.DataFrame) -> np.ndarray:
        features = self.features.all_features()
        X = df[features].values

        for _, phase_df in df.groupby("phase"):
            if len(phase_df) < 12 * 120:
                continue

            # 切り取る長さ
            max_length = len(phase_df) - 12 * 30 * 2
            crop_length = random.randint(max_length // 8, max_length // 2)

            # 切り取る位置
            start = random.randint(12 * 30, len(phase_df) - 12 * 30 - crop_length - 1)
            end = start + crop_length

            if phase_df["target"].mean() < 0.5:
                df_idx = random.randint(0, len(self.sleeping_dfs) - 1)
                if self.sleeping_dfs[df_idx].shape[0] > end + 12 * 30:
                    X[start:end] = self.sleeping_dfs[df_idx][features].values[start:end]
            else:
                df_idx = random.randint(0, len(self.awake_dfs) - 1)
                if self.awake_dfs[df_idx].shape[0] > end + 12 * 30:
                    X[start:end] = self.awake_dfs[df_idx][features].values[start:end]

        return X.astype(np.float32)
