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


def series_generate_features(train: pd.DataFrame, oof: pd.DataFrame) -> Tuple[pd.DataFrame, Features]:
    train = train.sort_values("step").reset_index(drop=True)

    # sensorの読み込み
    series_id = train["series_id"].unique()[0]
    path = os.path.join(CFG["dataset"]["step_csv_dir"], f"{series_id}.csv")
    sensor_df = pd.read_csv(path)
    if "target" in sensor_df.columns:
        sensor_df = sensor_df.drop("target", axis=1)
    if "event" in sensor_df.columns:
        sensor_df = sensor_df.drop("event", axis=1)

    # oofを結合
    oof["step"] = oof["step"].astype(int)
    sensor_df = sensor_df.merge(oof[["series_id", "step", "oof_stage2"]], on=["series_id", "step"], how="left")

    # oofを線形補間
    sensor_df["oof_stage2"] = sensor_df["oof_stage2"].interpolate(
        method="linear", limit_direction="both", limit_area=None
    )
    assert sensor_df["oof_stage2"].isna().sum() == 0

    # trainを結合
    train = train.merge(sensor_df, on=["series_id", "step"], how="right")
    train = train.sort_values("step").reset_index(drop=True)

    # 特徴生成
    features = Features()
    train["for_pred"] = train["target"].notna().astype(float)

    # event
    train["bin_event"] = train["event"].map({"onset": -1.0, "wakeup": 1.0})
    features.add_num_feature("bin_event")

    # 時刻
    timestamp = pd.to_datetime(train["timestamp"].values[0])
    total_seconds = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    train["total_seconds"] = (total_seconds + train.index * 5) % (24 * 60 * 60)  # [sec]
    train["minutes"] = train["total_seconds"] % (60 * 60)
    features.add_num_features(["total_seconds", "minutes"])

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

    columns += f_names
    columns += gb.columns.tolist()
    columns += ["for_pred"]

    # rolling
    dts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
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
    base_df = train[train["for_pred"] == 1]
    leave_index = train[train["for_pred"] == 1].index
    for dt, shift_features in shift_features_dic.items():
        for c in [0.25, 0.5, 1, 2, 5]:
            _dt = int(dt * c)
            if _dt == 0:
                continue
            shifted_index = leave_index + _dt
            created_index = leave_index[shifted_index < len(train)]
            shifted_index = shifted_index[shifted_index < len(train)]
            df = train.iloc[shifted_index].copy()
            f_names_plus = [f"{c}_shift_{_dt}" for c in shift_features]
            df.rename(columns=dict(zip(shift_features, f_names_plus)), inplace=True)
            df.set_index(created_index, inplace=True)
            base_df = base_df.merge(df[f_names_plus], left_index=True, right_index=True, how="left")
            features.add_num_features(f_names_plus)

            shifted_index = leave_index - _dt
            created_index = leave_index[shifted_index >= 0]
            shifted_index = shifted_index[shifted_index >= 0]
            df = train.iloc[shifted_index].copy()
            f_names_minus = [f"{c}_shift_{-_dt}" for c in shift_features]
            df.rename(columns=dict(zip(shift_features, f_names_minus)), inplace=True)
            df.set_index(created_index, inplace=True)
            base_df = base_df.merge(df[f_names_minus], left_index=True, right_index=True, how="left")
            features.add_num_features(f_names_minus)

            f_names_diff = [f"{c}_shift_{_dt}_diff" for c in shift_features]
            base_df[f_names_diff] = base_df[f_names_plus].values - base_df[f_names_minus].values
            features.add_num_features(f_names_diff)

    base_df = base_df.reset_index(drop=True)

    base_df = reduce_mem_usage(base_df)
    gc.collect()
    return base_df, features


def generate_features_for_group(group: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, Features]:
    train, oof = group
    df, features = series_generate_features(train, oof)
    return df, features


def generate_regressor_features(train: pd.DataFrame, oof: pd.date_range) -> Tuple[pd.DataFrame, Features]:
    groups = []
    for series_id in train["series_id"].unique():
        train_df = train[train["series_id"] == series_id].reset_index(drop=True)
        oof_df = oof[oof["series_id"] == series_id].reset_index(drop=True)
        groups.append([train_df, oof_df])
    with Pool(CFG["env"]["num_threads"]) as pool:
        results = list(tqdm(pool.imap(generate_features_for_group, groups), total=len(groups)))
    dfs, features_list = zip(*results)
    features = features_list[0]
    train = pd.concat(dfs).reset_index(drop=True)
    del dfs, results, groups
    gc.collect()

    return train, features
