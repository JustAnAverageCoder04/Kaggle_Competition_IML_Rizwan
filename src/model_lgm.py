"""LightGBM model for 30-day readmission.

The pipeline is deliberately thin: all shared behaviour (matrix construction,
grouped CV, validation, artifacts) lives in ``modeling_utils`` so this script
and ``model_xgb.py`` cannot drift apart.

Run::

    python src/model_lgm.py

Hyper-parameters below are the Optuna results produced by ``tuner_LGM.py``.
They were searched *with* ``subsample_freq=1`` and ``scale_pos_weight=9``, so
both are set here too - otherwise the searched configuration would not be the
configuration being fitted.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lightgbm import LGBMClassifier  # noqa: E402

from modeling_utils import (  # noqa: E402
    POS_WEIGHT,
    SEED,
    build_matrices,
    load_datasets,
    run_cv,
    save_artifacts,
)

TAG = "lgb"

PARAMS = {
    "n_estimators": 782,
    "learning_rate": 0.01698923124150896,
    "max_depth": 3,
    "num_leaves": 51,
    "min_child_samples": 42,
    "subsample": 0.7586645035200215,
    # Without this LightGBM ignores `subsample` entirely; the tuner set it to 1.
    "subsample_freq": 1,
    "colsample_bytree": 0.943889730654808,
    "reg_alpha": 0.008228516019554934,
    "reg_lambda": 0.7829702169807307,
    "scale_pos_weight": POS_WEIGHT,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def make_model() -> LGBMClassifier:
    """A fresh, unfitted estimator (one per fold)."""
    return LGBMClassifier(**PARAMS)


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
