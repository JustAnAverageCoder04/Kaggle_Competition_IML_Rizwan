"""Blend the LightGBM and XGBoost submissions into ``submission_ensemble.csv``.

Two fixes over the previous version:

* the two submissions are joined on ``encounter_id`` instead of being assumed to
  share the same row order, and a missing/duplicated row raises;
* the blend weight is optimised on the saved out-of-fold predictions when they
  are available, instead of being hard-coded at 0.4/0.6.  ``model_lgm.py`` used
  to be the only script saving OOF, so a weight could not be fitted at all.

Run after both model scripts::

    python src/the_ultimate_decider.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from schema import ID_COL, TARGET_COL  # noqa: E402
from modeling_utils import (  # noqa: E402
    CLEANED_DIR,
    MODELS_DIR,
    best_rank_blend_weight,
    rank_transform,
)

#: Fallback when no OOF vectors are on disk.
DEFAULT_WEIGHT = {"lgb": 0.4, "xgb": 0.6}
ENSEMBLE_NAME = "ensemble"


def _read_submission(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing submission at {path}")
    frame = pd.read_csv(path)
    missing = [c for c in (ID_COL, TARGET_COL) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicated {ID_COL}")
    return frame


def merge_submissions(lgb_sub: pd.DataFrame, xgb_sub: pd.DataFrame) -> pd.DataFrame:
    """Align both submissions on ``encounter_id``, preserving the LightGBM order."""
    lgb_ids = set(lgb_sub[ID_COL])
    xgb_ids = set(xgb_sub[ID_COL])
    if lgb_ids != xgb_ids:
        raise ValueError(
            "the submissions cover different encounter sets "
            f"(lgb={len(lgb_ids)}, xgb={len(xgb_ids)}, shared={len(lgb_ids & xgb_ids)})"
        )

    merged = lgb_sub[[ID_COL, TARGET_COL]].merge(
        xgb_sub[[ID_COL, TARGET_COL]],
        on=ID_COL,
        how="left",
        suffixes=("_lgb", "_xgb"),
        validate="one_to_one",
    )
    if len(merged) != len(lgb_sub) or len(merged) != len(xgb_sub):
        raise ValueError(
            "the submissions cover different encounter sets "
            f"(lgb={len(lgb_sub)}, xgb={len(xgb_sub)}, merged={len(merged)})"
        )
    if merged[[f"{TARGET_COL}_lgb", f"{TARGET_COL}_xgb"]].isna().any().any():
        raise ValueError("some encounters are missing a prediction from one model")
    return merged


def _load_oof(tag: str) -> np.ndarray | None:
    path = os.path.join(MODELS_DIR, f"oof_preds_{tag}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def _load_labels() -> pd.Series | None:
    path = os.path.join(CLEANED_DIR, "train_processed.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, usecols=[TARGET_COL])[TARGET_COL]
    except (ValueError, KeyError):
        return None


def choose_weight() -> tuple[float, str]:
    """Return (weight_for_lgb, description)."""
    oof_lgb = _load_oof("lgb")
    oof_xgb = _load_oof("xgb")
    labels = _load_labels()

    if oof_lgb is None or oof_xgb is None or labels is None:
        return DEFAULT_WEIGHT["lgb"], "default 0.4 / 0.6 (no OOF artifacts found)"

    if not (len(oof_lgb) == len(oof_xgb) == len(labels)):
        return (
            DEFAULT_WEIGHT["lgb"],
            "default 0.4 / 0.6 (OOF lengths do not match the training labels)",
        )

    weight, auc = best_rank_blend_weight(oof_lgb, oof_xgb, labels.to_numpy())
    return weight, f"OOF-optimised (blend ROC-AUC {auc:.4f})"


def main() -> None:
    lgb_sub = _read_submission(os.path.join(MODELS_DIR, "submission_lgb.csv"))
    xgb_sub = _read_submission(os.path.join(MODELS_DIR, "submission_xgb.csv"))
    merged = merge_submissions(lgb_sub, xgb_sub)

    weight, source = choose_weight()
    print(f"Blend weight for LightGBM: {weight:.2f} ({source})")

    ensemble = weight * rank_transform(merged[f"{TARGET_COL}_lgb"]) + (
        1.0 - weight
    ) * rank_transform(merged[f"{TARGET_COL}_xgb"])

    output = pd.DataFrame({ID_COL: merged[ID_COL].to_numpy(), TARGET_COL: ensemble})
    if output[TARGET_COL].isna().any():
        raise ValueError("the blended submission contains missing predictions")

    path = os.path.join(MODELS_DIR, f"submission_{ENSEMBLE_NAME}.csv")
    output.to_csv(path, index=False)
    print(f"Rank-averaged ensemble submission saved to: {path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
