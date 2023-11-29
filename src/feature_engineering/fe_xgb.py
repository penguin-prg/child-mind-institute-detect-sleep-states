import gc
import os
import pickle
import sys
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import polars as pl

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)
    CFG = yaml.safe_load(open(os.path.join(PACKAGE_DIR, "config.yaml"), "r"))

    cand_path = os.path.join(
        "/kaggle/output", CFG["patch_transformer_gru"]["execution"]["best_exp_id"], "next_cands.pkl"
    )
    with open(cand_path, "rb") as f:
        next_cands = pickle.load(f)

from utils.feature_contena import Features
from utils.pandas_utils import reduce_mem_usage


def series_generate_features(train: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    train = train.sort_values("step").reset_index(drop=True)
    train["anglez"] = train["anglez"].astype(np.float16)
    train["enmo"] = train["enmo"].astype(np.float16)
    train.drop(columns=["event"], inplace=True)

    features = Features()

    # 時刻
    timestamp = pd.to_datetime(train["timestamp"].values[0])
    total_seconds = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    train["total_seconds"] = (total_seconds + train.index * 5) % (24 * 60 * 60)  # [sec]
    train["minutes"] = train["total_seconds"] % (60 * 60)
    features.add_num_features(["total_seconds", "minutes"])
    del timestamp, total_seconds
    train.drop(columns=["timestamp"], inplace=True)
    gc.collect()

    columns = ["anglez", "enmo"]
    features.add_num_features(columns)

    # その人のその時刻での平均的な測定値
    gb = train.groupby("total_seconds")[columns].mean()
    gb.columns = [f"{c}_mean" for c in columns]
    train["anglez_mean"] = train["total_seconds"].map(gb["anglez_mean"])
    train["enmo_mean"] = train["total_seconds"].map(gb["enmo_mean"])
    features.add_num_features(gb.columns.tolist())

    # 30分間完全一致した回数
    train["same_count"] = 0.0
    step_per_day = 24 * 60 * 60 // 5
    n_day = int(len(train) // step_per_day + 1)
    for d in range(-n_day, n_day + 1):
        if d == 0:
            continue
        is_same = train["anglez"].diff(d * step_per_day).abs().rolling(12 * 30, center=True).sum() == 0
        train.loc[is_same, "same_count"] += 1
    train["same_count"] = train["same_count"].clip(0, 5)
    features.add_num_features(["same_count"])

    # diff
    f_names = [f"{c}_diff_abs" for c in columns]
    train[f_names] = train[columns].diff().abs()
    features.add_num_features(f_names)

    # diff abs clip
    train["anglez_diff_abs_clip5"] = train["anglez_diff_abs"].clip(0, 5)
    features.add_num_features(["anglez_diff_abs_clip5"])

    # 一定stepで集約
    series_id = train["series_id"].values[0]
    agg_freq = CFG["xgb_model"]["execution"]["agg_freq"]
    columns = features.all_features() + ["target", "step", "onset_target", "wakeup_target"]
    train_mean = train[columns].groupby(train["step"].values // agg_freq).mean()
    columns = features.all_features() + ["step"]
    train_std = train[columns].groupby(train["step"].values // agg_freq).std()
    train_std = train_std.drop(columns=["step"])
    train_std.columns = [f"{c}_std" for c in train_std.columns]
    train = pd.concat([train_mean, train_std], axis=1)
    features.add_num_features(train_std.columns.tolist())
    del train_mean, train_std
    gc.collect()
    train["series_id"] = series_id
    train["target"] = train["target"].round().astype(int)
    train = train.reset_index(drop=True)

    # rolling
    columns = [
        "enmo",
        # "anglez_diff_abs",
        "anglez_diff_abs_clip5",
        "enmo_std",
        # "anglez_diff_abs_std",
        "anglez_diff_abs_clip5_std",
        "same_count",
    ]
    dts = [1, 3, 5, 10, 30, 100]
    shift_features_dic = {}
    for dt in dts:
        shift_features = []

        f_names = [f"{c}_rolling_mean_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).mean()
        features.add_num_features(f_names)
        shift_features += f_names

        f_names = [f"{c}_rolling_std_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).std()
        features.add_num_features(f_names)
        shift_features += f_names

        f_names = [f"{c}_rolling_max_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).max()
        features.add_num_features(f_names)
        shift_features += f_names

        f_names = [f"{c}_rolling_median_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).median()
        features.add_num_features(f_names)
        shift_features += f_names

        f_names = [f"{c}_rolling_square_mean_{dt}" for c in columns]
        train[f_names] = (train[columns] ** 2).rolling(dt, center=True).mean()
        features.add_num_features(f_names)
        shift_features += f_names

        shift_features_dic[dt] = shift_features

    # shift
    for dt, shift_features in shift_features_dic.items():
        used = set()
        for c in [-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10]:
            _dt = int(dt * c)
            if _dt == 0 or _dt in used:
                continue
            used.add(_dt)
            f_names = [f"{c}_shift_{_dt}" for c in shift_features]
            train[f_names] = train[shift_features].shift(_dt)
            features.add_num_features(f_names)

    # next_candsにないstepは除外
    if series_id in next_cands:
        cands = next_cands[series_id]
        train["reduce_step"] = train["step"].astype(int)
        train = train[train["reduce_step"].isin(cands)]

    train = reduce_mem_usage(train)

    for f in features.all_features():
        if f == "total_seconds":
            continue
        train[f] = train[f].astype(np.float16)

    gc.collect()
    return train, features


def series_generate_features2(train: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    train = train.sort_values("step").reset_index(drop=True)
    features = Features()

    columns = ["is_longest_sleep_episode", "is_sleep_block"]
    features.add_num_features(columns)

    # 一定stepで集約
    series_id = train["series_id"].values[0]
    agg_freq = CFG["xgb_model"]["execution"]["agg_freq"]
    columns = features.all_features() + ["step"]
    train = train[columns].groupby(train["step"].values // agg_freq).mean()
    gc.collect()
    train = train.reset_index(drop=True)

    # rolling
    columns = ["is_longest_sleep_episode", "is_sleep_block"]
    dts = [1, 3, 5, 10, 30]
    shift_features_dic = {}
    for dt in dts:
        shift_features = []

        f_names = [f"{c}_rolling_mean_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).mean()
        features.add_num_features(f_names)
        shift_features += f_names

        shift_features_dic[dt] = shift_features

    # shift
    for dt, shift_features in shift_features_dic.items():
        used = set()
        for c in [-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10]:
            _dt = int(dt * c)
            if _dt == 0 or _dt in used:
                continue
            used.add(_dt)
            f_names = [f"{c}_shift_{_dt}" for c in shift_features]
            train[f_names] = train[shift_features].shift(_dt)
            features.add_num_features(f_names)

    # next_candsにないstepは除外
    if series_id in next_cands:
        cands = next_cands[series_id]
        train["reduce_step"] = train["step"].astype(int)
        train = train[train["reduce_step"].isin(cands)]

    train = reduce_mem_usage(train)

    for f in features.all_features():
        if f == "total_seconds":
            continue
        train[f] = train[f].astype(np.float16)

    gc.collect()
    return train, features


def read_and_generate_features(file: Tuple[str, str]) -> Tuple[pd.DataFrame, Features]:
    train = pd.read_parquet(file[0])
    fdf = pd.read_parquet(file[1], columns=["is_longest_sleep_episode", "is_sleep_block"])
    fdf["series_id"] = train["series_id"].values[0]
    fdf["step"] = train["step"]

    train2, features2 = series_generate_features2(fdf)
    del fdf
    gc.collect()
    train1, features1 = series_generate_features(train)
    del train
    gc.collect()

    assert all(train1["step"].values == train2["step"].values)
    train = pd.concat([train1, train2[features2.all_features()]], axis=1)
    features = Features()
    features.add_num_features(features1.all_features())
    features.add_num_features(features2.all_features())

    gc.collect()
    return train, features


def generate_features(files: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, Features]:
    with Pool(CFG["env"]["num_threads"]) as pool:
        results = list(tqdm(pool.imap(read_and_generate_features, files), total=len(files), desc="generate features"))
    dfs, features = zip(*results)
    train = pd.concat(dfs)
    features = features[0]

    return train, features
