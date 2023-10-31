import gc

import numpy as np
import pandas as pd
import lightgbm as lgbm
from tqdm import tqdm


def fit_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    features: list,
    params: dict,
    es_rounds=20,
    verbose=100,
    log=True,
):
    models = []
    oof = np.zeros(len(y), dtype=np.float64)

    if log:
        bar = tqdm(range(max(folds) + 1))
    else:
        bar = range(max(folds) + 1)
    for i in bar:
        if log:
            print(f"== fold {i} ==")
        trn_idx = folds != i
        val_idx = folds == i

        model = lgbm.LGBMRegressor(**params)
        model.fit(
            X[features][trn_idx],
            y[trn_idx],
            eval_set=[(X[features][val_idx], y[val_idx])],
            early_stopping_rounds=es_rounds,
            eval_metric="rmse",
            verbose=verbose,
        )
        valid_pred = model.predict(X[features][val_idx])
        oof[val_idx] = valid_pred
        models.append(model)

        del valid_pred
        gc.collect()

    return oof, models


def inference_lgb(models: list, feat_df: pd.DataFrame, pred_type: str = "regression"):
    assert pred_type in ["regression", "binary"]
    if pred_type == "regression":
        pred = np.array([model.predict(feat_df) for model in models]).mean(axis=0)
    return pred
