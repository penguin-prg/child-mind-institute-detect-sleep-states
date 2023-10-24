import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import sys
from typing import List


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
    def __init__(self, dfs: List[pd.DataFrame], mode: str, features: Features, patch_size: int, block_size: int):
        self.dfs = dfs
        self.mode = mode
        self.features = features
        self.patch_size = patch_size
        self.block_size = block_size

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df = self.dfs[index]

        block_size = self.block_size
        patch_size = self.patch_size
        n_feats = len(self.features.all_features())

        feats = df[self.features.all_features()].values.astype(np.float32)

        # padding
        if len(feats) < block_size:
            feats = np.concatenate([feats, np.zeros((block_size - len(feats), n_feats))], axis=0)
            ok_len = len(feats) // patch_size
            ng_len = block_size // patch_size - ok_len
            mask = np.concatenate([np.zeros(ok_len), np.ones(ng_len)], axis=0)
        else:
            mask = np.zeros(block_size // patch_size)
        mask = mask.astype(bool)

        feats = feats.reshape(block_size // patch_size, patch_size, n_feats)
        feats = feats.reshape(block_size // patch_size, patch_size * n_feats)

        if self.mode == "train":
            targets = df[["wakeup_target", "onset_target"]].values.astype(np.float32)
            targets = targets.reshape(-1, patch_size, 2).mean(axis=1)
            if len(targets) < feats.shape[0]:
                targets = np.concatenate([targets, np.zeros((feats.shape[0] - len(targets), 2))], axis=0)
            assert targets.shape == (block_size // patch_size, 2)
            return feats, mask, targets
        else:
            return feats, mask
