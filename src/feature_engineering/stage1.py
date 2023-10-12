import pandas as pd
import gc
import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
import yaml
import numpy as np

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)
    CFG = yaml.safe_load(open(os.path.join(PACKAGE_DIR, "config.yaml"), "r"))

from utils.pandas_utils import reduce_mem_usage
from utils.feature_contena import Features


def series_generate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    df = df.sort_values("step").reset_index(drop=True)

    features = Features()

    df["hour"] = df["timestamp"].str[11:13].astype(int)
    features.add_num_features(["hour"])

    df["lids"] = np.maximum(0.0, df["enmo"] - 0.02)
    df["lids"] = df["lids"].rolling(120, center=True, min_periods=1).agg("sum")
    df["lids"] = 100 / (df["lids"] + 1)
    df["lids"] = df["lids"].rolling(360, center=True, min_periods=1).agg("mean").astype(np.float32)
    features.add_num_features(["lids"])

    df["enmo"] = (df["enmo"] * 1000).astype(np.int16)
    df["anglez"] = df["anglez"].astype(np.int16)
    df["anglezdiffabs"] = df["anglez"].diff().abs().astype(np.float32)
    features.add_num_features(["enmo", "anglez", "anglezdiffabs"])

    for col in ["enmo", "anglez", "anglezdiffabs"]:
        # periods in seconds
        periods = [60, 360, 720]

        for n in periods:
            rol_args = {"window": int((n + 5) // 5), "min_periods": 1, "center": True}

            for agg in ["median", "mean", "max", "min", "var"]:
                df[f"{col}_{agg}_{n}"] = df[col].rolling(**rol_args).agg(agg).astype(np.float32).values
                gc.collect()
                features.add_num_features([f"{col}_{agg}_{n}"])

            if n == max(periods):
                df[f"{col}_mad_{n}"] = (
                    (df[col] - df[f"{col}_median_{n}"]).abs().rolling(**rol_args).median().astype(np.float32)
                )
                features.add_num_features([f"{col}_mad_{n}"])

            df[f"{col}_amplit_{n}"] = df[f"{col}_max_{n}"] - df[f"{col}_min_{n}"]
            df[f"{col}_amplit_{n}_min"] = df[f"{col}_amplit_{n}"].rolling(**rol_args).min().astype(np.float32).values
            features.add_num_features([f"{col}_amplit_{n}", f"{col}_amplit_{n}_min"])

            df[f"{col}_diff_{n}_max"] = df[f"{col}_max_{n}"].diff().abs().rolling(**rol_args).max().astype(np.float32)
            df[f"{col}_diff_{n}_mean"] = df[f"{col}_max_{n}"].diff().abs().rolling(**rol_args).mean().astype(np.float32)
            features.add_num_features([f"{col}_diff_{n}_max", f"{col}_diff_{n}_mean"])

            gc.collect()

    for f in features.num_features():
        df[f] = df[f].astype(np.float32)

    # 一定stepで集約
    series_id = df["series_id"].values[0]
    agg_freq = CFG["feature"]["agg_freq"]
    columns = features.all_features() + ["target", "step"]
    df = df[columns].groupby(df["step"].values // agg_freq).mean()
    df["series_id"] = series_id
    df["target"] = df["target"].round().astype(int)

    df = reduce_mem_usage(df)
    gc.collect()
    return df, features


def read_and_generate_features(file: str) -> Tuple[pd.DataFrame, Features]:
    train = pd.read_csv(file)
    train, features = series_generate_features(train)
    return train, features


def generate_1st_stage_features(files: List[str]) -> Tuple[pd.DataFrame, Features]:
    with Pool(CFG["env"]["num_threads"]) as pool:
        results = list(tqdm(pool.imap(read_and_generate_features, files), total=len(files), desc="generate features"))
    dfs, features = zip(*results)
    train = pd.concat(dfs)
    features = features[0]

    return train, features
