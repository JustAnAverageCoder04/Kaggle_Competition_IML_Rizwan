"""XGBoost model for 30-day readmission.

Shares the matrix construction and the grouped CV with ``model_lgm.py`` through
``modeling_utils``, so the two models see exactly the same features and the
same folds.  In particular the missing-value policy is now identical: XGBoost
keeps native ``NaN`` and so does LightGBM, instead of one model receiving
``-999`` while the other received ``NaN``.

Run::

    python src/model_xgb.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from xgboost import XGBClassifier  # noqa: E402

from modeling_utils import (  # noqa: E402
    POS_WEIGHT,
    SEED,
    build_matrices,
    load_datasets,
    run_cv,
    save_artifacts,
)

TAG = "xgb"

PARAMS = {
    "n_estimators": 502,
    "learning_rate": 0.039685922357215006,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.6299899700841038,
    "colsample_bytree": 0.8418153389411714,
    "reg_alpha": 4.949713849530675,
    "reg_lambda": 0.0024758797143213554,
    "scale_pos_weight": POS_WEIGHT,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
}


def make_model() -> XGBClassifier:
    """A fresh, unfitted estimator (one per fold)."""
    return XGBClassifier(**PARAMS)


def main() -> None:
    train_df, test_df = load_datasets()
    matrices = build_matrices(train_df, test_df)

    oof, test_preds, fold_scores, overall_auc = run_cv(
        make_model, matrices, tag=TAG
    )
    save_artifacts(
        TAG, matrices, oof, test_preds, fold_scores, overall_auc, params=PARAMS
    )


if __name__ == "__main__":
    main()
