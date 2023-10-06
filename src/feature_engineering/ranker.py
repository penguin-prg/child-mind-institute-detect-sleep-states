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
    train = train.sort_values(["step", "level"]).reset_index(drop=True)

    features = Features()
    features.add_num_features(["level", "score"])

    # event
    train["bin_event"] = train["event"].map({"onset": 0.0, "wakeup": 1.0})
    features.add_num_feature("bin_event")

    # 時刻
    train["total_seconds"] = train.index * 5 * CFG["feature"]["agg_freq"] % (24 * 60 * 60)
    features.add_num_feature("total_seconds")

    # 分
    features.add_num_feature("minutes")

    # gap
    train["sub_gap"] = train["sub_step"] - train["step"]
    train["sub_minutes"] = (train["minutes"] + train["sub_gap"] * 12) % (60 * 60)
    train["before_gap"] = train["sub_step_before_modify"] - train["step"]
    train["before_minutes"] = (train["minutes"] + train["before_gap"] * 12) % (60 * 60)
    features.add_num_features(["sub_gap", "sub_minutes", "before_gap", "before_minutes", "oof_regressor"])

    columns = ["oof_stage2"]

    # その人のその時刻での平均的な測定値
    gb = train.groupby("total_seconds")[columns].mean()
    gb.columns = [f"{c}_mean" for c in columns]
    train["oof_stage2_mean"] = train["total_seconds"].map(gb["oof_stage2_mean"])
    features.add_num_features(gb.columns.tolist())

    # diff
    f_names = [f"{c}_diff_abs" for c in columns]
    train[f_names] = train[columns].diff().abs()
    features.add_num_features(f_names)

    # 予測対象か
    train["for_pred"] = train["target"].notna().astype(float)

    columns += f_names
    columns += gb.columns.tolist()
    columns += ["for_pred"]

    # rolling
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

    train = train.dropna(subset=["target"]).reset_index(drop=True)

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
