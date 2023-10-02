import pandas as pd
import gc
import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
import yaml

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)
    CFG = yaml.safe_load(open(os.path.join(PACKAGE_DIR, "config.yaml"), "r"))

from utils.pandas_utils import reduce_mem_usage
from utils.feature_contena import Features


def series_generate_features(train: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    features = Features()

    # 時刻
    timestamp = pd.to_datetime(train["timestamp"].values[0])
    total_seconds = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    train["total_seconds"] = (total_seconds + train.index * 5) % (24 * 60 * 60)  # [sec]
    features.add_num_feature("total_seconds")

    columns = ["anglez", "enmo"]

    # その人のその時刻での平均的な測定値
    gb = train.groupby("total_seconds")[columns].mean()
    gb.columns = [f"{c}_mean" for c in columns]
    train["anglez_mean"] = train["total_seconds"].map(gb["anglez_mean"])
    train["enmo_mean"] = train["total_seconds"].map(gb["enmo_mean"])
    features.add_num_features(gb.columns.tolist())
    columns += gb.columns.tolist()

    # diff
    f_names = [f"{c}_diff_abs" for c in columns]
    train[f_names] = train[columns].diff().abs()
    features.add_num_features(f_names)
    columns += f_names

    # rolling
    dts = [10, 50, 100, 1000]
    for dt in dts:
        f_names = [f"{c}_rolling_mean_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).mean()
        features.add_num_features(f_names)

        f_names = [f"{c}_rolling_std_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).std()
        features.add_num_features(f_names)

        f_names = [f"{c}_rolling_max_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).max()
        features.add_num_features(f_names)

        f_names = [f"{c}_rolling_min_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).min()
        features.add_num_features(f_names)

        f_names = [f"{c}_rolling_median_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).median()
        features.add_num_features(f_names)

        f_names = [f"{c}_rolling_square_mean_{dt}" for c in columns]
        train[f_names] = (train[columns] ** 2).rolling(dt, center=True).mean()
        features.add_num_features(f_names)

    # 一定stepで集約
    series_id = train["series_id"].values[0]
    agg_freq = CFG["feature"]["agg_freq"]
    columns = features.all_features() + ["target_wakeup", "target_onset", "step"]
    train = train[columns].groupby(train["step"].values // agg_freq).mean()
    train["series_id"] = series_id
    train[["target_wakeup", "target_onset"]] = train[["target_wakeup", "target_onset"]].round().astype(int)

    train = reduce_mem_usage(train)
    gc.collect()
    return train, features


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
