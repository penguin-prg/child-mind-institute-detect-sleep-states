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
    train = train.sort_values("step").reset_index(drop=True)
    features = Features()

    # event
    features.add_num_features(["for_wakeup", "for_onset"])

    # score
    features.add_num_features(["wakeup_oof", "onset_oof", "wakeup_score", "onset_score"])

    # gap
    train["wakeup_reduced_score"] = train["wakeup_oof"] - train["wakeup_score"]
    train["onset_reduced_score"] = train["onset_oof"] - train["onset_score"]
    features.add_num_features(["wakeup_reduced_score", "onset_reduced_score"])

    # その人のその時刻での平均的な測定値
    columns = ["wakeup_oof", "onset_oof"]
    train["time"] = train.index % (60 * 24)
    gb = train.groupby("time")[columns].mean()
    gb.columns = [f"{c}_gb_mean" for c in columns]
    for c in gb.columns:
        train[c] = train["time"].map(gb[c])
    features.add_num_features(gb.columns.tolist())

    # diff
    columns = ["wakeup_oof", "onset_oof"]  # , "wakeup_score", "onset_score"]
    f_names = [f"{c}_diff" for c in columns]
    train[f_names] = train[columns].diff()
    features.add_num_features(f_names)

    # rolling
    columns += f_names
    dts = [1, 2, 3, 5, 10, 50]
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

        f_names = [f"{c}_rolling_min_{dt}" for c in columns]
        train[f_names] = train[columns].rolling(dt, center=True).min()
        features.add_num_features(f_names)
        shift_features += f_names

        f_names = [f"{c}_rolling_square_mean_{dt}" for c in columns]
        train[f_names] = (train[columns] ** 2).rolling(dt, center=True).mean()
        features.add_num_features(f_names)
        shift_features += f_names

        shift_features_dic[dt] = shift_features

    # shift
    for dt, shift_features in shift_features_dic.items():
        for c in [0.5, 1, 2]:
            _dt = int(dt * c)
            if _dt == 0:
                continue
            f_names_plus = [f"{c}_shift_{_dt}" for c in shift_features]
            train[f_names_plus] = train[shift_features].shift(_dt)
            features.add_num_features(f_names_plus)

            f_names_minus = [f"{c}_shift_{-_dt}" for c in shift_features]
            train[f_names_minus] = train[shift_features].shift(-_dt)
            features.add_num_features(f_names_minus)

            f_names_diff = [f"{c}_shift_{_dt}_diff" for c in shift_features]
            train[f_names_diff] = train[f_names_plus].values - train[f_names_minus].values
            features.add_num_features(f_names_diff)

    train = pd.concat(
        [
            train[train["for_wakeup"] == True],
            train[train["for_onset"] == True],
        ]
    ).reset_index(drop=True)

    train = reduce_mem_usage(train)
    gc.collect()
    return train, features


def generate_features_for_group(group: Tuple[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Features]:
    _, df = group
    df, features = series_generate_features(df)
    return df, features


def generate_ranker_features(train: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    groups = list(train.groupby("series_id"))
    with Pool(CFG["env"]["num_threads"]) as pool:
        results = list(tqdm(pool.imap(generate_features_for_group, groups), total=len(groups)))
    dfs, features_list = zip(*results)
    features = features_list[0]
    train = pd.concat(dfs).reset_index(drop=True)
    del dfs, results, groups
    gc.collect()

    return train, features
