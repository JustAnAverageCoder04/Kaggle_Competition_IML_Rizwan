"""Regression tests for the modelling layer (``src/modeling_utils.py`` and the
decider's submission alignment).  Synthetic frames only - ``data/`` is never
touched.

Runs with pytest *or* standalone: ``python tests/run_tests.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import modeling_utils as mu  # noqa: E402
import the_ultimate_decider as decider  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_frames(n_patients: int = 12, per_patient: int = 5, seed: int = 0):
    """A tiny train/test pair shaped like the cleaned files."""
    rng = np.random.default_rng(seed)
    n = n_patients * per_patient
    patient = np.repeat(np.arange(n_patients), per_patient)

    train = pd.DataFrame(
        {
            "encounter_id": np.arange(1000, 1000 + n),
            "patient_id": patient,
            "hospital_id": patient % 3,
            "los_days": rng.gamma(2.0, 2.0, size=n),
            "lab_mean_x": rng.normal(size=n),
        }
    )
    train.loc[rng.random(n) < 0.1, "lab_mean_x"] = np.nan
    train["readmit_30d"] = (rng.random(n) < 0.25).astype(int)

    test = train.drop(columns=["readmit_30d"]).copy()
    test["encounter_id"] = np.arange(5000, 5000 + n)
    return train, test


def _expect_value_error(callable_, needle: str) -> None:
    try:
        callable_()
    except ValueError as exc:
        assert needle in str(exc), f"unexpected message: {exc}"
        return
    raise AssertionError(f"expected ValueError mentioning {needle!r}")


def _temp_dir() -> str:
    """Workspace-local scratch directory.

    ``tempfile.mkdtemp`` is avoided on purpose: it creates the directory with a
    restrictive ACL that the harness file sandbox refuses to write through.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"_test_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def fast_model():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=5,
        num_leaves=3,
        min_child_samples=2,
        subsample=1.0,
        verbose=-1,
        random_state=0,
    )


# --------------------------------------------------------------------------- #
# matrix contract
# --------------------------------------------------------------------------- #
def test_identifiers_never_reach_the_model():
    train, test = make_frames()
    matrices = mu.build_matrices(train, test)

    assert matrices.n_features == 2
    forbidden = {"encounter_id", "patient_id", "hospital_id"}
    assert not forbidden & set(matrices.X.columns)
    assert not forbidden & set(matrices.X_test.columns)
    assert "readmit_30d" not in matrices.X.columns


def test_groups_default_to_patient_not_hospital():
    train, test = make_frames()
    matrices = mu.build_matrices(train, test)
    assert matrices.groups.name == "patient_id"
    assert (matrices.groups.to_numpy() == train["patient_id"].to_numpy()).all()


def test_folds_never_share_a_patient():
    from sklearn.model_selection import StratifiedGroupKFold

    train, test = make_frames()
    matrices = mu.build_matrices(train, test)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)

    for train_idx, val_idx in splitter.split(
        matrices.X, matrices.y, groups=matrices.groups
    ):
        fitted = set(matrices.groups.iloc[train_idx])
        validated = set(matrices.groups.iloc[val_idx])
        assert not fitted & validated


def test_train_test_schema_mismatch_fails_loudly():
    train, test = make_frames()
    test = test.drop(columns=["lab_mean_x"])
    _expect_value_error(lambda: mu.build_matrices(train, test), "schema mismatch")


def test_leaky_feature_name_is_rejected():
    train, test = make_frames()
    train["readmit_risk_score"] = 0.1
    test["readmit_risk_score"] = 0.1
    _expect_value_error(
        lambda: mu.build_matrices(train, test), "readmit_risk_score"
    )


def test_non_binary_target_is_rejected():
    train, test = make_frames()
    train.loc[train.index[0], "readmit_30d"] = 2
    _expect_value_error(lambda: mu.build_matrices(train, test), "0/1")


def test_build_xy_matches_build_matrices():
    train, test = make_frames()
    matrices = mu.build_matrices(train, test)
    X, y, groups = mu.build_xy(train)
    assert list(X.columns) == list(matrices.X.columns)
    assert len(X) == len(matrices.X) == len(y) == len(groups)


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #
def test_run_cv_covers_every_row_and_returns_valid_probabilities():
    train, test = make_frames()
    matrices = mu.build_matrices(train, test)

    oof, test_preds, fold_scores, overall = mu.run_cv(
        fast_model, matrices, tag="unit", n_splits=2
    )

    assert oof.shape == (len(matrices.X),)
    assert np.isfinite(oof).all()
    assert (oof >= 0).all() and (oof <= 1).all()
    assert test_preds.shape == (len(matrices.X_test),)
    assert np.isfinite(test_preds).all()
    assert len(fold_scores) == 2
    assert 0.0 <= overall <= 1.0


def test_save_artifacts_keeps_test_row_order():
    train, test = make_frames()
    matrices = mu.build_matrices(train, test)
    oof, test_preds, fold_scores, overall = mu.run_cv(
        fast_model, matrices, tag="unit", n_splits=2
    )

    directory = _temp_dir()
    try:
        mu.save_artifacts(
            "unit",
            matrices,
            oof,
            test_preds,
            fold_scores,
            overall,
            params={"n_estimators": 5},
            models_dir=directory,
        )
        submission = pd.read_csv(os.path.join(directory, "submission_unit.csv"))
        with open(os.path.join(directory, "metrics_unit.json"), encoding="utf-8") as fh:
            metrics = json.load(fh)
        assert os.path.exists(os.path.join(directory, "oof_preds_unit.npy"))
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    assert submission["encounter_id"].tolist() == matrices.test_ids.tolist()
    assert len(submission) == len(matrices.test_ids)
    assert metrics["n_features"] == matrices.n_features
    assert np.isfinite(submission["readmit_30d"]).all()


# --------------------------------------------------------------------------- #
# blending
# --------------------------------------------------------------------------- #
def test_rank_blend_beats_or_matches_the_better_single_model():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    strong = y + rng.normal(0.0, 0.5, size=200)
    weak = y + rng.normal(0.0, 3.0, size=200)

    weight, auc = mu.best_rank_blend_weight(strong, weak, y)

    single_strong = roc_auc_score(y, mu.rank_transform(strong))
    single_weak = roc_auc_score(y, mu.rank_transform(weak))

    assert 0.0 <= weight <= 1.0
    assert auc >= max(single_strong, single_weak) - 1e-12
    assert weight > 0.5, "the informative model should dominate the blend"


def test_rank_blend_reaches_a_perfect_separation():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    informative = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.85, 0.25, 0.75, 0.3])
    inverted = 1.0 - informative

    weight, auc = mu.best_rank_blend_weight(informative, inverted, y)

    assert 0.0 <= weight <= 1.0
    assert auc == 1.0


def test_blend_weight_rejects_length_mismatch():
    _expect_value_error(
        lambda: mu.best_rank_blend_weight(np.zeros(5), np.zeros(4), np.zeros(5)),
        "length mismatch",
    )


def test_decider_aligns_on_encounter_id_not_row_order():
    lgb = pd.DataFrame(
        {"encounter_id": [1, 2, 3], "readmit_30d": [0.1, 0.2, 0.3]}
    )
    xgb = pd.DataFrame(
        {"encounter_id": [3, 1, 2], "readmit_30d": [0.9, 0.8, 0.7]}
    )

    merged = decider.merge_submissions(lgb, xgb)

    assert merged["encounter_id"].tolist() == [1, 2, 3]
    assert merged["readmit_30d_xgb"].tolist() == [0.8, 0.7, 0.9]


def test_decider_rejects_partial_overlap():
    lgb = pd.DataFrame({"encounter_id": [1, 2], "readmit_30d": [0.1, 0.2]})
    xgb = pd.DataFrame({"encounter_id": [2, 3], "readmit_30d": [0.8, 0.7]})
    _expect_value_error(
        lambda: decider.merge_submissions(lgb, xgb), "different encounter sets"
    )
