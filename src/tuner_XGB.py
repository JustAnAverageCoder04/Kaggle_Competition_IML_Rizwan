"""Optuna hyper-parameter search for the XGBoost model.

Mirrors ``tuner_LGM.py``: same matrix construction, same patient-grouped folds,
same ``scale_pos_weight`` as ``model_xgb.py``, and native ``NaN`` (XGBoost
handles it, so there is no ``-999`` sentinel that the final model does not use).

Run::

    python src/tuner_XGB.py
    READMIT_N_TRIALS=100 python src/tuner_XGB.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import optuna  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from modeling_utils import (  # noqa: E402
    MODELS_DIR,
    N_SPLITS,
    POS_WEIGHT,
    SEED,
    build_xy,
    load_datasets,
)

N_TRIALS = int(os.environ.get("READMIT_N_TRIALS", "50"))
TAG = "xgb"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: "optuna.Trial") -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": POS_WEIGHT,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": -1,
    }


def make_objective(X, y, groups, n_splits: int = N_SPLITS):
    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED
        )
        oof = np.full(len(X), np.nan, dtype="float64")

        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        if np.isnan(oof).any():
            raise RuntimeError("cross-validation did not cover every training row")
        return float(roc_auc_score(y, oof))

    return objective


def main() -> None:
    train_df, _ = load_datasets()
    X, y, groups = build_xy(train_df)
    print(f"Tuning XGBoost on {X.shape[1]} features, grouped by {groups.name!r}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(X, y, groups),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n--- Tuning Complete ---")
    print(f"Best OOF ROC-AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"best_params_{TAG}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"best_value": study.best_value, "params": study.best_params},
            handle,
            indent=2,
            default=str,
        )
    print(f"Saved best parameters to {out_path}")


if __name__ == "__main__":
    main()
