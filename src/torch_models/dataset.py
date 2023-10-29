import os
import sys
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
