import xgboost as xgb
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc

from typing import Dict, List, Optional
from xgboost.core import Booster


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


def plot_importances(models: List[Booster], save_path: Optional[str] = None) -> Dict[str, float]:
    """特徴量の重要度をプロットする"""
    # 重要度の統計量
    all_importances = {f: [] for f in models[0].feature_names}
    for model in models:
        importance = model.get_score(importance_type="gain")
        for f, imp in importance.items():
            all_importances[f].append(imp)
    average_importances = {
        feature: (sum(values) / len(values)) if len(values) > 0 else 0 for feature, values in all_importances.items()
    }
    sorted_features = sorted(average_importances.keys(), key=lambda x: average_importances[x], reverse=True)
    sorted_values = [all_importances[feature] for feature in sorted_features]

    # グラフの描画
    plt.figure(figsize=(max(int(len(importance) / 5), 10), 6))
    plt.boxplot(sorted_values, labels=sorted_features, showfliers=False)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.grid()
    plt.title("Feature Importance using Gain")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

    return average_importances
