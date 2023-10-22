import pandas as pd
import gc
import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
import yaml
import pickle

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)
    CFG = yaml.safe_load(open(os.path.join(PACKAGE_DIR, "config.yaml"), "r"))

    cand_path = os.path.join("/kaggle/output", CFG["1st_stage"]["execution"]["best_exp_id"], "next_cands.pkl")
    with open(cand_path, "rb") as f:
        next_cands = pickle.load(f)

from utils.pandas_utils import reduce_mem_usage
from utils.feature_contena import Features


def series_generate_features(train: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    train = train.sort_values("step").reset_index(drop=True)

    features = Features()

    # 時刻
    timestamp = pd.to_datetime(train["timestamp"].values[0])
    total_seconds = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    train["total_seconds"] = (total_seconds + train.index * 5) % (24 * 60 * 60)  # [sec]
    train["minutes"] = train["total_seconds"] % (60 * 60)
    features.add_num_features(["total_seconds", "minutes"])

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
    features.add_num_features(["same_count"])

    # diff
    f_names = [f"{c}_diff_abs" for c in columns]
    train[f_names] = train[columns].diff().abs()
    features.add_num_features(f_names)

    # 一定stepで集約
    series_id = train["series_id"].values[0]
    agg_freq = CFG["2nd_stage"]["execution"]["agg_freq"]
    columns = features.all_features() + ["target", "step", "onset_target", "wakeup_target"]
    train = train[columns].groupby(train["step"].values // agg_freq).mean()
    train["series_id"] = series_id
    train["target"] = train["target"].round().astype(int)
    train = train.reset_index(drop=True)

    # rolling
    columns = ["enmo"] + ["anglez_diff_abs"]
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
    gc.collect()
    return train, features


def read_and_generate_features(file: str) -> Tuple[pd.DataFrame, Features]:
    train = pd.read_parquet(file)
    train, features = series_generate_features(train)
    gc.collect()
    return train, features


def generate_2nd_stage_features(files: List[str]) -> Tuple[pd.DataFrame, Features]:
    with Pool(CFG["env"]["num_threads"]) as pool:
        results = list(tqdm(pool.imap(read_and_generate_features, files), total=len(files), desc="generate features"))
    dfs, features = zip(*results)
    train = pd.concat(dfs)
    features = features[0]

    return train, features
