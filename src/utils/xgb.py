import lightgbm as lgbm
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc


def fit_xgb(
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
        X_train, y_train = X[features][trn_idx], y[trn_idx]
        X_valid, y_valid = X[features][val_idx], y[val_idx]

        model = lgbm.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=es_rounds,
            eval_metric="logloss",
            verbose=verbose,
        )

        train_pred = model.predict_proba(X_train)[:, 1]
        valid_pred = model.predict_proba(X_valid)[:, 1]

        oof[val_idx] = valid_pred
        models.append(model)
        del X_train, X_valid, y_train, y_valid, train_pred, valid_pred
        gc.collect()

    return oof, models


def inference_xgb(models: list, feat_df: pd.DataFrame, pred_type: str = "regression"):
    pred = np.array([model.predict_proba(feat_df)[:, 1] for model in models])
    pred = np.mean(pred, axis=0)
    return pred
