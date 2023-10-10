import gc
import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map


def process_series(args):
    series_id, df, score_duration, min_sleep_step, window_step = args

    df = df.reset_index(drop=True)
    df["raw_oof"] = df["oof"]
    df["oof"] = df["oof"].rolling(window_step, center=True).mean().fillna(1)

    # イベントの時刻
    wakeup_index = np.array([i for i in range(1, len(df["oof"])) if df["oof"][i - 1] < 0.5 and df["oof"][i] >= 0.5])
    onset_index = np.array([i for i in range(1, len(df["oof"])) if df["oof"][i - 1] > 0.5 and df["oof"][i] <= 0.5])

    # 就寝時間が短すぎる場合は除外
    sub_idxs = []
    for i in range(len(wakeup_index)):
        idx_diff = wakeup_index[i] - (onset_index[i] if i < len(onset_index) else 0)
        if idx_diff >= min_sleep_step or idx_diff < 0:
            sub_idxs.append(i)
    wakeup_index = wakeup_index[sub_idxs]
    onset_index = onset_index[[idx for idx in sub_idxs if idx < len(onset_index)]]

    if len(wakeup_index) == 0 or len(onset_index) == 0:
        return pd.DataFrame()

    # confidence
    wakeup_scores = []
    for idx in wakeup_index:
        wakeup_scores.append(
            df["oof"].values[min(idx + score_duration, len(df) - 1)] - df["oof"].values[max(0, idx - score_duration)]
        )
    onset_scores = []
    for idx in onset_index:
        onset_scores.append(
            -df["oof"].values[min(idx + score_duration, len(df) - 1)] + df["oof"].values[max(0, idx - score_duration)]
        )

    # 提出形式
    sub = pd.concat(
        [
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "step": df["step"].values[wakeup_index],
                    "event": "wakeup",
                    "score": wakeup_scores,
                }
            )
            if len(wakeup_index) > 0
            else pd.DataFrame(),
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "step": df["step"].values[onset_index],
                    "event": "onset",
                    "score": onset_scores,
                }
            )
            if len(onset_index) > 0
            else pd.DataFrame(),
        ]
    )
    return sub


def post_process(
    train: pd.DataFrame, score_duration_step: int = 12, min_sleep_step: int = 12 * 30, window_step: int = 12 * 24
) -> pd.DataFrame:
    iterable = [
        (series_id, group, score_duration_step, min_sleep_step, window_step)
        for series_id, group in train.groupby("series_id")
    ]

    dfs = process_map(process_series, iterable, max_workers=cpu_count(), chunksize=1, desc="Processing series")

    sub = pd.concat(dfs).reset_index(drop=True)
    sub = sub.sort_values(["series_id", "step"]).reset_index(drop=True)

    del dfs
    gc.collect()
    return sub
