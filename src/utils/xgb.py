import xgboost as xgb
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc


def custom_bce(preds, dtrain):
    # ソフトラベルを取得
    labels = dtrain.get_label()
    # シグモイド関数で予測を[0,1]の間に制約
    preds = 1.0 / (1.0 + np.exp(-preds))
    # 損失の計算
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


def custom_bce_eval(preds, dtrain):
    # ソフトラベルを取得
    labels = dtrain.get_label()
    # シグモイド関数で予測を[0,1]の間に制約
    preds = 1.0 / (1.0 + np.exp(-preds))
    # 評価指標の計算
    err = -np.mean(labels * np.log(preds) + (1.0 - labels) * np.log(1.0 - preds))
    return "custom_bce", err


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
            obj=custom_bce,
            feval=custom_bce_eval,
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
