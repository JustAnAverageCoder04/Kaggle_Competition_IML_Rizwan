"""Shared training utilities for the readmission models.

Every model script and every tuner goes through this module, so the feature
matrix, the cross-validation grouping, the missing-value policy and the
artifact naming cannot drift apart again.

Key contracts enforced here:

* identifiers (``encounter_id`` / ``patient_id`` / ``hospital_id``) are dropped
  from the design matrix - ``patient_id`` used to reach ``.fit()``;
* train and test must have exactly the same columns and dtypes - a mismatch
  raises instead of being papered over with ``reindex(fill_value=0)``;
* folds are grouped by ``patient_id`` so a patient never appears in both the
  fitted data and the validation data;
* missing values are left as ``NaN`` for both libraries, which have native
  handling, instead of one model getting ``-999`` and the other ``NaN``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from schema import (  # noqa: E402  (import must follow the sys.path tweak)
    HOSPITAL_COL,
    ID_COL,
    ID_COLS,
    PATIENT_COL,
    TARGET_COL,
    check_binary_target,
    leaky_columns,
    require_columns,
)

BASE_DIR = os.path.dirname(_HERE)
CLEANED_DIR = os.environ.get(
    "READMIT_CLEANED_DIR", os.path.join(BASE_DIR, "data", "cleaned")
)
MODELS_DIR = os.environ.get("READMIT_MODELS_DIR", os.path.join(BASE_DIR, "models"))

#: Grouping key for the CV splitter. Patient-level grouping is the default
#: because prior-utilisation features are patient-level: grouping by hospital
#: still lets the same person sit on both sides of a fold.
GROUP_COL = os.environ.get("READMIT_GROUP_COLUMN", PATIENT_COL)

N_SPLITS = int(os.environ.get("READMIT_N_SPLITS", "5"))
SEED = int(os.environ.get("READMIT_SEED", "42"))

#: ``scale_pos_weight`` used by both the tuners and the final models, so the
#: tuned hyper-parameters are actually optimal for the model being fitted.
POS_WEIGHT = float(os.environ.get("READMIT_POS_WEIGHT", "9"))


@dataclass
class Matrices:
    """Everything one model needs, built once and validated once."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    X_test: pd.DataFrame
    test_ids: pd.Series

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_datasets(cleaned_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed train/test files written by ``garbage_duty``."""
    cleaned_dir = cleaned_dir or CLEANED_DIR
    paths = {
        "train": os.path.join(cleaned_dir, "train_processed.csv"),
        "test": os.path.join(cleaned_dir, "test_processed.csv"),
    }
    for label, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} split not found at {path} - run `python src/garbage_duty.py` first"
            )
    return pd.read_csv(paths["train"]), pd.read_csv(paths["test"])


# --------------------------------------------------------------------------- #
# matrix construction
# --------------------------------------------------------------------------- #
def _numeric_block(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep numeric/boolean columns, dropping anything else with a warning."""
    block = frame.select_dtypes(include=[np.number, "bool"]).copy()
    dropped = [column for column in frame.columns if column not in block.columns]
    if dropped:
        warnings.warn(
            f"{name}: dropping non-numeric column(s) {dropped}",
            RuntimeWarning,
            stacklevel=3,
        )
    numeric = block.select_dtypes(include=[np.number])
    if numeric.shape[1]:
        block[numeric.columns] = numeric.replace([np.inf, -np.inf], np.nan)
    block.columns = [str(column) for column in block.columns]
    return block


def _check_feature_names(X: pd.DataFrame) -> None:
    leaked = leaky_columns(X, ignore=set(ID_COLS))
    if leaked:
        raise ValueError(
            "outcome-like column(s) reached the model matrix: "
            f"{leaked}. Fix the feature builder or drop them explicitly."
        )
    identifiers = [c for c in X.columns if c in ID_COLS]
    if identifiers:
        raise ValueError(f"identifier column(s) reached the model matrix: {identifiers}")
    if X.shape[1] == 0:
        raise ValueError("the model matrix has no features")


def build_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    group_col: str | None = None,
) -> Matrices:
    """Build aligned design matrices, the label and the CV groups."""
    group_col = group_col or GROUP_COL
    require_columns(train_df, "train", [ID_COL, TARGET_COL])
    require_columns(test_df, "test", [ID_COL])

    y = check_binary_target(train_df[TARGET_COL])

    if group_col in train_df.columns:
        groups = train_df[group_col]
    elif HOSPITAL_COL in train_df.columns:
        warnings.warn(
            f"grouping column {group_col!r} missing; falling back to {HOSPITAL_COL!r}, "
            "which does not keep patients out of the validation folds",
            RuntimeWarning,
            stacklevel=2,
        )
        groups = train_df[HOSPITAL_COL]
    else:
        raise ValueError(
            f"no grouping column available (looked for {group_col!r} and {HOSPITAL_COL!r})"
        )
    if groups.isna().any():
        raise ValueError(f"grouping column {group_col!r} contains missing values")

    drop = [ID_COL, TARGET_COL, PATIENT_COL, HOSPITAL_COL]
    train_raw = train_df.drop(columns=[c for c in drop if c in train_df.columns])
    test_raw = test_df.drop(columns=[c for c in drop if c in test_df.columns])

    if list(train_raw.columns) != list(test_raw.columns):
        only_train = [c for c in train_raw.columns if c not in test_raw.columns]
        only_test = [c for c in test_raw.columns if c not in train_raw.columns]
        raise ValueError(
            "train/test feature schema mismatch; rebuild data/cleaned with "
            "`python src/garbage_duty.py`. "
            f"train-only={only_train[:5]} test-only={only_test[:5]} "
            f"order_matches={set(train_raw.columns) == set(test_raw.columns)}"
        )

    X = _numeric_block(train_raw, "train")
    X_test = _numeric_block(test_raw, "test")
    if list(X.columns) != list(X_test.columns):
        only_train = [c for c in X.columns if c not in X_test.columns]
        only_test = [c for c in X_test.columns if c not in X.columns]
        raise ValueError(
            "train/test dtypes diverged after numeric filtering; "
            f"train-only={only_train[:5]} test-only={only_test[:5]}"
        )

    _check_feature_names(X)

    empty_in_test = [
        column
        for column in X_test.columns
        if X_test[column].isna().all() and not X[column].isna().all()
    ]
    if empty_in_test:
        warnings.warn(
            "column(s) present in train but entirely missing in test: "
            f"{empty_in_test[:10]}",
            RuntimeWarning,
            stacklevel=2,
        )

    constant = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
    if constant:
        warnings.warn(
            f"{len(constant)} constant feature(s), e.g. {constant[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )

    return Matrices(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        test_ids=test_df[ID_COL].reset_index(drop=True),
    )


def build_xy(
    train_df: pd.DataFrame, *, group_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix, label and groups for a run that needs no test frame.

    Used by the tuners so they optimise on byte-identical features and folds to
    the final models; reusing the train frame keeps a single code path.
    """
    matrices = build_matrices(train_df, train_df, group_col=group_col)
    return matrices.X, matrices.y, matrices.groups


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #
def _positive_proba(model, X: pd.DataFrame, label: str) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError(f"{label}: predict_proba returned shape {proba.shape}")
    positive = np.asarray(proba[:, 1], dtype="float64")
    if positive.shape != (len(X),):
        raise RuntimeError(
            f"{label}: got {len(positive)} probabilities for {len(X)} rows"
        )
    if not np.isfinite(positive).all():
        raise RuntimeError(f"{label}: non-finite probabilities")
    if positive.min() < 0.0 or positive.max() > 1.0:
        raise RuntimeError(f"{label}: probabilities outside [0, 1]")
    return positive


def run_cv(
    make_model: Callable[[], object],
    matrices: Matrices,
    *,
    tag: str = "model",
    n_splits: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """Grouped CV; returns (oof, test_preds, fold_scores, overall_auc)."""
    n_splits = n_splits or N_SPLITS
    seed = SEED if seed is None else seed

    X, y, groups, X_test = (
        matrices.X,
        matrices.y,
        matrices.groups,
        matrices.X_test,
    )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan, dtype="float64")
    test_preds = np.zeros(len(X_test), dtype="float64")
    fold_scores: list[float] = []

    print(
        f"[{tag}] {n_splits}-fold grouped CV | {X.shape[1]} features | "
        f"groups={matrices.groups.name} | n_train={len(X)} n_test={len(X_test)}"
    )

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        expected = getattr(model, "n_features_in_", X.shape[1])
        if expected != X.shape[1]:
            raise RuntimeError(
                f"[{tag}] fold {fold}: model was fitted on {expected} features, "
                f"frame has {X.shape[1]}"
            )

        fold_oof = _positive_proba(model, X.iloc[val_idx], f"[{tag}] fold {fold} val")
        fold_test = _positive_proba(model, X_test, f"[{tag}] fold {fold} test")

        oof[val_idx] = fold_oof
        test_preds += fold_test / n_splits

        score = float(roc_auc_score(y.iloc[val_idx], fold_oof))
        fold_scores.append(score)
        print(f"[{tag}] fold {fold}/{n_splits} roc_auc={score:.4f}")

    if np.isnan(oof).any():
        raise RuntimeError(
            f"[{tag}] {int(np.isnan(oof).sum())} row(s) never received an out-of-fold "
            "prediction; the splitter did not cover the training set"
        )

    overall = float(roc_auc_score(y, oof))
    print(
        f"[{tag}] grouped OOF roc_auc={overall:.4f} "
        f"(fold mean {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f})"
    )
    return oof, test_preds, fold_scores, overall


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def save_artifacts(
    tag: str,
    matrices: Matrices,
    oof: np.ndarray,
    test_preds: np.ndarray,
    fold_scores: Iterable[float],
    overall_auc: float,
    params: dict | None = None,
    *,
    models_dir: str | None = None,
) -> dict:
    """Persist submission, OOF vector and metrics for one model."""
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    if len(test_preds) != len(matrices.test_ids):
        raise ValueError(
            f"[{tag}] {len(test_preds)} predictions for {len(matrices.test_ids)} test rows"
        )

    submission = pd.DataFrame(
        {
            ID_COL: matrices.test_ids.to_numpy(),
            TARGET_COL: np.asarray(test_preds, dtype="float64"),
        }
    )
    if submission[ID_COL].duplicated().any():
        raise ValueError(f"[{tag}] duplicate encounter_id in the submission")

    submission_path = os.path.join(models_dir, f"submission_{tag}.csv")
    submission.to_csv(submission_path, index=False)
    np.save(os.path.join(models_dir, f"oof_preds_{tag}.npy"), np.asarray(oof, dtype="float64"))

    fold_scores = [float(score) for score in fold_scores]
    metrics = {
        "model": tag,
        "n_features": matrices.n_features,
        "n_train": int(len(matrices.X)),
        "n_test": int(len(matrices.X_test)),
        "group_column": str(matrices.groups.name),
        "n_splits": len(fold_scores),
        "seed": SEED,
        "fold_roc_auc": fold_scores,
        "mean_fold_roc_auc": float(np.mean(fold_scores)),
        "std_fold_roc_auc": float(np.std(fold_scores)),
        "overall_oof_roc_auc": float(overall_auc),
        "params": params or {},
    }
    metrics_path = os.path.join(models_dir, f"metrics_{tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    print(f"[{tag}] submission -> {submission_path}")
    print(f"[{tag}] oof        -> {os.path.join(models_dir, f'oof_preds_{tag}.npy')}")
    print(f"[{tag}] metrics    -> {metrics_path}")
    return metrics


# --------------------------------------------------------------------------- #
# blending
# --------------------------------------------------------------------------- #
def rank_transform(values) -> np.ndarray:
    """Percentile ranks, which makes two differently calibrated models comparable."""
    from scipy.stats import rankdata

    array = np.asarray(values, dtype="float64")
    if len(array) == 0:
        raise ValueError("cannot rank an empty array")
    return rankdata(array) / len(array)


def best_rank_blend_weight(
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    y,
    *,
    step: float = 0.01,
) -> tuple[float, float]:
    """Grid-search the weight of ``oof_a`` in a rank blend, scored by ROC-AUC."""
    oof_a = np.asarray(oof_a, dtype="float64")
    oof_b = np.asarray(oof_b, dtype="float64")
    y = np.asarray(y)
    if not (len(oof_a) == len(oof_b) == len(y)):
        raise ValueError(
            f"length mismatch: oof_a={len(oof_a)} oof_b={len(oof_b)} y={len(y)}"
        )

    ranks_a = rank_transform(oof_a)
    ranks_b = rank_transform(oof_b)

    best_weight, best_auc = 0.5, float("-inf")
    for weight in np.arange(0.0, 1.0 + 1e-9, step):
        auc = roc_auc_score(y, weight * ranks_a + (1.0 - weight) * ranks_b)
        if auc > best_auc:
            best_weight, best_auc = float(weight), float(auc)
    return best_weight, best_auc
