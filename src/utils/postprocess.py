from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import gc
from typing import Optional
from scipy.interpolate import interp1d

RANGE = 917
COEFF = 11
EXP = 5

if True:
    PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(PACKAGE_DIR)
    CFG = yaml.safe_load(open(os.path.join(PACKAGE_DIR, "config.yaml"), "r"))


def post_process(train: pd.DataFrame, output_dir: Optional[str] = None) -> pd.DataFrame:
    dfs = []
    for series_id, df in tqdm(train.groupby("series_id"), desc="post process"):
        df = df.reset_index(drop=True)
        df["raw_oof"] = df["oof"]
        df["oof"] = df["oof"].rolling(24, center=True).mean().fillna(1)

        # イベントの時刻
        wakeup_index = np.array([i for i in range(1, len(df["oof"])) if df["oof"][i - 1] < 0.5 and df["oof"][i] >= 0.5])
        onset_index = np.array([i for i in range(1, len(df["oof"])) if df["oof"][i - 1] > 0.5 and df["oof"][i] <= 0.5])

        # 就寝時間が短すぎる場合は除外
        sub_idxs = []
        for i in range(len(wakeup_index)):
            idx_diff = wakeup_index[i] - (onset_index[i] if i < len(onset_index) else 0)
            if idx_diff >= 15 or idx_diff < 0:
                sub_idxs.append(i)
        wakeup_index = wakeup_index[sub_idxs]
        onset_index = onset_index[[idx for idx in sub_idxs if idx < len(onset_index)]]

        if len(wakeup_index) == 0 or len(onset_index) == 0:
            continue

        # confidence
        SCORE_DURATION = 12
        wakeup_scores = []
        for idx in wakeup_index:
            wakeup_scores.append(
                df["oof"].values[min(idx + SCORE_DURATION, len(df) - 1)]
                - df["oof"].values[max(0, idx - SCORE_DURATION)]
            )
        onset_scores = []
        for idx in onset_index:
            onset_scores.append(
                -df["oof"].values[min(idx + SCORE_DURATION, len(df) - 1)]
                + df["oof"].values[max(0, idx - SCORE_DURATION)]
            )

        # 提出形式
        sub = pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "key_step": df["step"].values[wakeup_index],
                        "event": "wakeup",
                        "score": wakeup_scores,
                    }
                )
                if len(wakeup_index) > 0
                else pd.DataFrame(),
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "key_step": df["step"].values[onset_index],
                        "event": "onset",
                        "score": onset_scores,
                    }
                )
                if len(onset_index) > 0
                else pd.DataFrame(),
            ]
        )
        sub["step"] = (sub["key_step"] - CFG["feature"]["agg_freq"] // 2).astype(int)
        dfs.append(sub)

        if output_dir is not None:
            _, axs = plt.subplots(3, 1, figsize=(20, 7))
            axs[0].plot(df["oof"])
            axs[0].plot(df["target"])
            axs[0].scatter(wakeup_index, [0.5 for _ in wakeup_index], c="red")
            axs[0].scatter(onset_index, [0.5 for _ in onset_index], c="green")
            df = pd.read_csv(f"{CFG['dataset']['step_csv_dir']}/{df['series_id'].values[0]}.csv")
            axs[1].plot(df["enmo"])
            axs[2].plot(df["anglez"])
            plt.suptitle(f"series_id: {series_id}")
            plt.tight_layout()
            path = os.path.join(output_dir, f"series_graph/{series_id}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close()
    sub = pd.concat(dfs).reset_index(drop=True)
    sub = sub.sort_values(["series_id", "step"]).reset_index(drop=True)

    del dfs
    gc.collect()
    return sub


def dynamic_range_nms(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamic-Range NMS

    Parameters
    ----------
    df : pd.DataFrame
        単一のseries_idに対する提出形式
    """
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    used = []
    used_scores = []
    reduce_rate = np.ones(df["step"].max() + 1000)
    for _ in range(min(len(df), 1000)):
        df["reduced_score"] = df["score"] / reduce_rate[df["step"]]
        best_score = df["reduced_score"].max()
        best_idx = df["reduced_score"].idxmax()
        best_step = df.loc[best_idx, "step"]
        used.append(best_idx)
        used_scores.append(best_score)

        for r in range(1, int(RANGE)):
            reduce = ((RANGE - r) / RANGE) ** EXP * COEFF
            reduce_rate[best_step + r] += reduce
            if best_step - r >= 0:
                reduce_rate[best_step - r] += reduce
        reduce_rate[best_step] = 1e10
    df = df.iloc[used].copy()
    df["reduced_score"] = used_scores
    return df
