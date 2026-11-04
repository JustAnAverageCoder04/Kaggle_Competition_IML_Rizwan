"""Regression tests for the feature engineering stage (``src/garbage_duty.py``).

Runs with pytest *or* standalone: ``python tests/run_tests.py``.  No fixtures,
no network and no access to ``data/`` - every test builds synthetic tables.
"""

from __future__ import annotations

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

import garbage_duty as gd  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _ts(*values):
    return pd.to_datetime(list(values))


def synthetic_raw() -> dict:
    """Four encounters: a patient with two admissions, one with one, one with none."""
    return {
        "index_encounters": pd.DataFrame(
            {
                "encounter_id": [1, 2, 3, 4],
                "patient_id": [7, 7, 8, 9],
                "hospital_id": [10, 11, 10, 11],
                "admit_dt": _ts(
                    "2024-01-01 08:00", "2024-06-01 09:00",
                    "2024-02-01 10:00", "2024-03-01 11:00",
                ),
                "discharge_dt": _ts(
                    "2024-01-05", "2024-06-05", "2024-02-04", "2024-03-03"
                ),
            }
        ),
        "patients": pd.DataFrame(
            {"patient_id": [7, 8, 9], "birth_year": [1950, 1980, 1990]}
        ),
        "hospitals": pd.DataFrame({"hospital_id": [10, 11]}),
        "diagnoses": pd.DataFrame(
            {
                "encounter_id": [1, 1, 2, 3],
                "icd10_code": ["I50.9", "E11.9", "J18.9", "I50.9"],
                "is_primary": ["Y", "N", "Y", "Y"],
                "present_on_admission": ["Y", "N", "Y", "N"],
            }
        ),
        "icd10_lookup": pd.DataFrame(
            {
                "icd10_code": ["I50.9", "E11.9", "J18.9"],
                "chapter": ["Circulatory", "Endocrine", "Respiratory"],
                "chronic_flag": [1, 1, 0],
            }
        ),
        "procedures": pd.DataFrame(
            {
                "encounter_id": [1, 2],
                "code": ["PROC1", "PROC2"],
                "code_type": ["surgical", "medical"],
            }
        ),
        "medications": pd.DataFrame(
            {
                "encounter_id": [1, 1, 2],
                "drug_class": ["diuretic", "statin", "antibiotic"],
                "route": ["oral", "oral", "iv"],
                "continued_at_discharge": ["Y", "N", "Y"],
            }
        ),
        "lab_results": pd.DataFrame(
            {
                "encounter_id": [1, 2],
                "lab_name": ["Creatinine", "Glucose"],
                "value": ["1.1", "7.0"],
                "unit": ["mg/dL", "mmol/L"],
                "abnormal_flag": ["H", "N"],
            }
        ),
        "vitals": pd.DataFrame(
            {
                "encounter_id": [1, 2],
                "vital_type": ["TEMP", "TEMP"],
                "value": ["98.6", "37.0"],
            }
        ),
        "unit_transfers": pd.DataFrame(
            {
                "encounter_id": [1],
                "unit": ["ICU"],
                "in_dt": _ts("2024-01-01 09:00"),
                "out_dt": _ts("2024-01-02 09:00"),
            }
        ),
        # patient 7 has one visit before encounter 1, then two visits that fall
        # between encounter 1 and encounter 2.
        "prior_utilization": pd.DataFrame(
            {
                "patient_id": [7, 7, 7, 8],
                "visit_dt": _ts("2023-11-01", "2024-04-15", "2024-05-20", "2023-12-01"),
                "visit_type": ["ED", "ED", "ED", "inpatient"],
            }
        ),
        "discharge_planning": pd.DataFrame(
            {
                "encounter_id": [1, 2, 3, 4],
                "followup_scheduled_days": [7, 14, 3, 5],
                "home_health_referral": ["Y", "N", "N", "N"],
                "education_done": ["Yes", "No", "Yes", "Yes"],
            }
        ),
    }


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


# --------------------------------------------------------------------------- #
# leakage regressions
# --------------------------------------------------------------------------- #
def test_prior_visits_are_scoped_to_one_admission():
    """Encounter 1 must not see visits that happened before encounter 2."""
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")

    assert master.loc[1, "prior_visits_90d"] == 1
    assert master.loc[2, "prior_visits_90d"] == 2
    assert master.loc[1, "total_prior_visits"] == 1
    assert master.loc[2, "total_prior_visits"] == 3


def test_no_feature_looks_into_the_future():
    master = gd.build_features(synthetic_raw(), verbose=False)
    days = master["days_since_last_prior_visit"].dropna()
    assert (days >= 0).all(), f"{int((days < 0).sum())} negative gaps"


def test_encounter_without_history_gets_zero_not_missing():
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")
    assert master.loc[4, "prior_visits_90d"] == 0
    assert master.loc[4, "total_prior_visits"] == 0


def test_outcome_like_source_column_is_rejected():
    raw = synthetic_raw()
    raw["index_encounters"] = raw["index_encounters"].assign(
        readmitted_30d=[0, 1, 0, 1]
    )
    _expect_value_error(
        lambda: gd.build_features(raw, verbose=False), "readmitted_30d"
    )


def test_prior_visit_after_last_admission_still_raises_or_is_excluded():
    """A visit dated after the final admission must not create a negative gap."""
    raw = synthetic_raw()
    raw["prior_utilization"] = pd.concat(
        [
            raw["prior_utilization"],
            pd.DataFrame(
                {
                    "patient_id": [9],
                    "visit_dt": _ts("2024-09-01"),
                    "visit_type": ["ED"],
                }
            ),
        ],
        ignore_index=True,
    )
    master = gd.build_features(raw, verbose=False).set_index("encounter_id")
    assert master.loc[4, "prior_visits_90d"] == 0


# --------------------------------------------------------------------------- #
# feature correctness
# --------------------------------------------------------------------------- #
def test_temperature_is_converted_to_celsius():
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")
    assert abs(master.loc[1, "vital_mean_TEMP"] - 37.0) < 1e-6
    assert abs(master.loc[2, "vital_mean_TEMP"] - 37.0) < 1e-6


def test_lab_units_are_harmonised():
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")
    # 7.0 mmol/L glucose -> ~126.1 mg/dL
    assert master.loc[2, "lab_mean_Glucose"] > 100.0
    assert master.loc[2, "lab_abnormal_rate_Glucose"] == 0.0


def test_absent_event_tables_mean_zero_not_nan():
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")
    assert master.loc[2, "unit_hours_ICU"] == 0.0
    assert master.loc[2, "total_labs"] >= 1
    assert master.loc[4, "total_labs"] == 0.0


def test_binary_columns_are_not_summed_as_strings():
    master = gd.build_features(synthetic_raw(), verbose=False).set_index("encounter_id")
    # encounter 1: one primary (Y) and one secondary (N) diagnosis, one POA=Y/one N
    assert master.loc[1, "primary_diagnoses"] == 1.0
    assert master.loc[1, "poa_yes"] == 1.0
    assert master.loc[1, "poa_no"] == 1.0
    assert master.loc[1, "medications_continued_at_discharge"] == 1.0


def test_identifiers_are_kept_but_never_one_hot_encoded():
    master = gd.build_features(synthetic_raw(), verbose=False)
    assert {"encounter_id", "patient_id", "hospital_id"} <= set(master.columns)
    assert not [c for c in master.columns if c.startswith("patient_id_")]
    assert "admit_dt" not in master.columns
    assert "discharge_dt" not in master.columns


def test_high_cardinality_categorical_is_dropped_visibly():
    frame = pd.DataFrame(
        {
            "encounter_id": range(40),
            "free_text_note": [f"note {i}" for i in range(40)],
        }
    )
    encoded = gd.encode_categoricals(frame, verbose=False)
    assert "free_text_note" not in encoded.columns


def test_low_cardinality_categorical_is_one_hot_encoded():
    frame = pd.DataFrame(
        {"encounter_id": [1, 2, 3, 4], "sex": ["F", "M", "F", None]}
    )
    encoded = gd.encode_categoricals(frame, verbose=False)
    assert "sex" not in encoded.columns
    assert "sex_M" in encoded.columns
    assert "sex_Missing" in encoded.columns


# --------------------------------------------------------------------------- #
# train/test contract
# --------------------------------------------------------------------------- #
def test_split_writes_matching_feature_columns():
    raw = synthetic_raw()
    raw["train"] = pd.DataFrame({"encounter_id": [1, 2], "readmit_30d": [0, 1]})
    raw["test"] = pd.DataFrame({"encounter_id": [3, 4]})

    master = gd.build_features(raw, verbose=False)
    directory = _temp_dir()
    try:
        gd.split_and_write(
            master, raw["train"], raw["test"], directory, verbose=False
        )
        train = pd.read_csv(os.path.join(directory, "train_processed.csv"))
        test = pd.read_csv(os.path.join(directory, "test_processed.csv"))
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    assert "readmit_30d" not in test.columns
    assert [c for c in train.columns if c != "readmit_30d"] == list(test.columns)
    assert len(train) == 2 and len(test) == 2


def test_split_rejects_a_missing_train_encounter():
    raw = synthetic_raw()
    raw["train"] = pd.DataFrame({"encounter_id": [1, 99], "readmit_30d": [0, 1]})
    raw["test"] = pd.DataFrame({"encounter_id": [3]})

    master = gd.build_features(raw, verbose=False)
    directory = _temp_dir()
    try:
        _expect_value_error(
            lambda: gd.split_and_write(
                master, raw["train"], raw["test"], directory
            ),
            "lost",
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)
