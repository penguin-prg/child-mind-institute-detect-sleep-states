import xgboost as xgb
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
        dtrain = xgb.DMatrix(X[features][trn_idx], label=y[trn_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(X[features][val_idx], label=y[val_idx], enable_categorical=True)
        evallist = [(dvalid, "eval")]

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000000,
            evals=evallist,
            early_stopping_rounds=es_rounds,
            verbose_eval=verbose,
        )
        valid_pred = model.predict(dvalid)
        oof[val_idx] = valid_pred
        models.append(model)

        del dtrain, dvalid, valid_pred
        gc.collect()

    return oof, models


def inference_xgb(models: list, feat_df: pd.DataFrame, pred_type: str = "regression"):
    assert pred_type in ["regression", "binary"]
    dtrain = xgb.DMatrix(feat_df, enable_categorical=True)
    if pred_type == "regression":
        pred = np.array([model.predict(dtrain) for model in models]).mean(axis=0)
    return pred