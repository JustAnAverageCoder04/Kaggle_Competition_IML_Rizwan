"""Feature engineering for the 30-day readmission task.

Builds exactly one feature row per index encounter and writes the two modelling
files consumed by ``src/model_lgm.py`` and ``src/model_xgb.py``.

Leakage rules enforced here:

* every feature must be knowable at the discharge of the index encounter;
  rows stamped after that discharge are dropped by ``_temporal_guard``;
* prior utilisation is aggregated **per encounter** and cut off at the index
  admission, so a later admission's history cannot leak backwards;
* identifiers are kept for grouping/auditing but never encoded as features;
* a source column whose name looks like an outcome aborts the run instead of
  silently becoming a feature.

Run::

    python src/garbage_duty.py

Directories can be overridden with ``READMIT_RAW_DIR`` / ``READMIT_CLEANED_DIR``,
which is how the tests drive the pipeline against synthetic fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from schema import (  # noqa: E402  (import must follow the sys.path tweak)
    HOSPITAL_COL,
    ID_COLS,
    ID_COL,
    PATIENT_COL,
    TARGET_COL,
    leaky_columns,
    require_columns,
)

BASE_DIR = os.path.dirname(_HERE)
RAW_DIR = os.environ.get("READMIT_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
CLEANED_DIR = os.environ.get("READMIT_CLEANED_DIR", os.path.join(BASE_DIR, "data", "cleaned"))

#: Categoricals with more levels than this are dropped instead of one-hot encoded.
MAX_CATEGORICAL_LEVELS = 25

#: Window used for the "recent prior utilisation" features.
PRIOR_WINDOW_DAYS = 90

_RAW_TABLES = (
    "train",
    "test",
    "index_encounters",
    "patients",
    "hospitals",
    "diagnoses",
    "icd10_lookup",
    "procedures",
    "medications",
    "lab_results",
    "vitals",
    "unit_transfers",
    "prior_utilization",
    "discharge_planning",
)

#: Timestamp columns used to bound each event table at the index discharge.
_TIME_COLUMNS = {
    "diagnoses": ("diagnosis_dt", "onset_dt", "recorded_dt"),
    "procedures": ("procedure_dt", "proc_dt", "start_dt", "end_dt", "recorded_dt"),
    "medications": ("start_dt", "end_dt", "order_dt", "admin_dt"),
    "lab_results": ("result_dt", "collected_dt", "charttime", "lab_dt"),
    "vitals": ("recorded_dt", "charttime", "vital_dt"),
}

#: Known abnormal-flag vocabularies. Anything else is reported, not guessed.
_ABNORMAL_MAP = {
    "N": 0.0,
    "H": 1.0,
    "L": 1.0,
    "HH": 1.0,
    "LL": 1.0,
    "HU": 1.0,
    "LU": 1.0,
    "A": 1.0,
}

#: Aggregates that mean "none of this happened" and must be 0, not NaN.
_ZERO_FILL_PREFIXES = (
    "total_",
    "unique_",
    "primary_",
    "chronic_",
    "poa_",
    "procedure_code_count_",
    "procedure_type_count_",
    "medication_class_count_",
    "medication_route_count_",
    "lab_count_",
    "lab_abnormal_count_",
    "vital_count_",
    "unit_transfer_count_",
    "unit_hours_",
    "prior_visit_type_count_",
)
_ZERO_FILL_COLUMNS = ("prior_visits_90d", "medications_continued_at_discharge")

#: Columns that are never features (identifiers are handled separately).
_DROP_AFTER_BUILD = ("admit_dt", "discharge_dt", "last_prior_visit_dt", "attending_id")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def _read(raw: dict, key: str) -> pd.DataFrame:
    if key not in raw:
        raise KeyError(f"raw table {key!r} missing; available: {sorted(raw)}")
    return raw[key].copy()


def load_raw(raw_dir: str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read every source table. Never mutates the files on disk."""
    tables: dict[str, pd.DataFrame] = {}
    for name in _RAW_TABLES:
        path = os.path.join(raw_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"expected source table at {path}")
        tables[name] = pd.read_csv(path)
    return tables


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_binary(series: pd.Series) -> pd.Series:
    """Map ``Y/N``, ``Yes/No``, ``true/false`` and ``0/1`` onto floats.

    Unknown text raises instead of quietly turning into ``NaN`` (which the old
    ``.sum()`` aggregations would then have silently dropped).
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "y": 1.0,
            "yes": 1.0,
            "true": 1.0,
            "t": 1.0,
            "1": 1.0,
            "n": 0.0,
            "no": 0.0,
            "false": 0.0,
            "f": 0.0,
            "0": 0.0,
        }
    )
    if mapped.isna().all() and series.notna().any():
        sample = list(pd.unique(series.dropna()))[:5]
        raise ValueError(f"cannot interpret {series.name!r} as binary; saw {sample}")
    return mapped.astype("float64")


def _code_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(".", "", regex=False).str.upper()


def count_by_category(
    frame: pd.DataFrame, key: str, category: str, prefix: str
) -> pd.DataFrame:
    """One zero-filled count column per category level, with stable ordering."""
    values = frame[[key, category]].dropna(subset=[category]).copy()
    values[category] = values[category].astype(str)
    counts = values.groupby([key, category]).size().unstack(fill_value=0)
    counts = counts.reindex(sorted(counts.columns), axis=1)
    counts.columns = [f"{prefix}_{value}" for value in counts.columns]
    return counts.reset_index()


def _temporal_guard(
    frame: pd.DataFrame,
    name: str,
    discharge_dt: pd.Series,
    candidates,
    *,
    enabled: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Drop rows recorded after the discharge of their own index encounter.

    Only applied when the table actually carries a timestamp column; when it
    does not, the assumption that the table is discharge-bounded is reported so
    it stays visible instead of becoming folklore.
    """
    if not enabled:
        return frame

    present = [c for c in candidates if c in frame.columns]
    if not present:
        _log(f"  [temporal] {name}: no timestamp column; assuming discharge-bounded rows.", verbose)
        return frame

    for column in present:
        stamp = _to_datetime(frame[column])
        cutoff = frame[ID_COL].map(discharge_dt)
        resolvable = cutoff.notna().mean()
        after = (stamp.notna() & cutoff.notna() & (stamp > cutoff)).fillna(False)
        dropped = int(after.sum())
        if dropped:
            frame = frame.loc[~after].copy()
            _log(f"  [temporal] {name}.{column}: dropped {dropped} row(s) after discharge.", verbose)
        if resolvable < 0.5:
            _log(
                f"  [temporal] {name}: only {resolvable:.0%} of rows map to an index "
                "encounter; the discharge cutoff may be incomplete.",
                verbose,
            )
    return frame


# --------------------------------------------------------------------------- #
# per-table feature builders
# --------------------------------------------------------------------------- #
def add_encounter_features(
    encounters: pd.DataFrame, patients: pd.DataFrame, hospitals: pd.DataFrame
) -> pd.DataFrame:
    """Encounter/patient/hospital level features (all known at discharge)."""
    require_columns(
        encounters,
        "index_encounters",
        [ID_COL, PATIENT_COL, HOSPITAL_COL, "admit_dt", "discharge_dt"],
    )

    encounters["admit_dt"] = _to_datetime(encounters["admit_dt"])
    encounters["discharge_dt"] = _to_datetime(encounters["discharge_dt"])
    encounters["length_of_stay_days"] = (
        encounters["discharge_dt"] - encounters["admit_dt"]
    ).dt.total_seconds() / 86400.0

    negative_los = encounters["length_of_stay_days"] < 0
    if negative_los.any():
        print(
            f"  [warn] {int(negative_los.sum())} encounter(s) discharged before "
            "admission; LOS set to NaN."
        )
        encounters.loc[negative_los, "length_of_stay_days"] = np.nan

    encounters["admission_month"] = encounters["admit_dt"].dt.month
    encounters["admission_dayofweek"] = encounters["admit_dt"].dt.dayofweek
    encounters["admission_hour"] = encounters["admit_dt"].dt.hour

    patients = patients.drop_duplicates(subset=[PATIENT_COL])
    patients["birth_year"] = pd.to_numeric(patients["birth_year"], errors="coerce")
    birth_years = patients.set_index(PATIENT_COL)["birth_year"]
    encounters["age_at_admission"] = (
        encounters["admit_dt"].dt.year - encounters[PATIENT_COL].map(birth_years)
    )

    hospitals = hospitals.drop_duplicates(subset=[HOSPITAL_COL])
    volume = encounters.groupby(HOSPITAL_COL).size().rename("hospital_encounter_volume")
    hospitals = hospitals.merge(volume, on=HOSPITAL_COL, how="left")
    hospitals["hospital_encounter_volume"] = hospitals["hospital_encounter_volume"].fillna(0.0)

    master = encounters.merge(patients, on=PATIENT_COL, how="left", suffixes=("", "_patient"))
    master = master.merge(hospitals, on=HOSPITAL_COL, how="left")
    return master


def aggregate_diagnoses(diagnoses: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        diagnoses,
        "diagnoses",
        [ID_COL, "icd10_code", "is_primary", "present_on_admission"],
    )
    require_columns(lookup, "icd10_lookup", ["icd10_code", "chapter", "chronic_flag"])

    diagnoses["_code_key"] = _code_key(diagnoses["icd10_code"])
    lookup = lookup.copy()
    lookup["_code_key"] = _code_key(lookup["icd10_code"])
    lookup = lookup[["_code_key", "chapter", "chronic_flag"]].drop_duplicates("_code_key")
    diagnoses = diagnoses.merge(lookup, on="_code_key", how="left")

    diagnoses["chronic_flag"] = _to_binary(diagnoses["chronic_flag"])
    diagnoses["_primary"] = _to_binary(diagnoses["is_primary"])
    diagnoses["_poa"] = _to_binary(diagnoses["present_on_admission"])
    diagnoses["_poa_no"] = 1.0 - diagnoses["_poa"]

    aggregated = diagnoses.groupby(ID_COL).agg(
        total_diagnoses=("icd10_code", "count"),
        unique_diagnoses=("icd10_code", "nunique"),
        primary_diagnoses=("_primary", "sum"),
        chronic_diagnoses=("chronic_flag", "sum"),
        poa_yes=("_poa", "sum"),
        poa_no=("_poa_no", "sum"),
    ).reset_index()
    return aggregated.merge(
        count_by_category(diagnoses, ID_COL, "chapter", "diagnosis_chapter_count"),
        on=ID_COL,
        how="left",
    )


def aggregate_procedures(procedures: pd.DataFrame) -> pd.DataFrame:
    require_columns(procedures, "procedures", [ID_COL, "code"])

    aggregated = procedures.groupby(ID_COL).agg(
        total_procedures=("code", "size"),
        unique_procedure_codes=("code", "nunique"),
    ).reset_index()
    aggregated = aggregated.merge(
        count_by_category(procedures, ID_COL, "code", "procedure_code_count"),
        on=ID_COL,
        how="left",
    )
    if "code_type" in procedures.columns:
        aggregated = aggregated.merge(
            count_by_category(procedures, ID_COL, "code_type", "procedure_type_count"),
            on=ID_COL,
            how="left",
        )
    return aggregated


def aggregate_medications(medications: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        medications, "medications", [ID_COL, "drug_class", "continued_at_discharge"]
    )

    medications["continued_at_discharge"] = _to_binary(medications["continued_at_discharge"])
    aggregated = medications.groupby(ID_COL).agg(
        total_medications=("drug_class", "size"),
        unique_drug_classes=("drug_class", "nunique"),
        medications_continued_at_discharge=("continued_at_discharge", "sum"),
    ).reset_index()
    aggregated = aggregated.merge(
        count_by_category(medications, ID_COL, "drug_class", "medication_class_count"),
        on=ID_COL,
        how="left",
    )
    if "route" in medications.columns:
        aggregated = aggregated.merge(
            count_by_category(medications, ID_COL, "route", "medication_route_count"),
            on=ID_COL,
            how="left",
        )
    return aggregated


def _harmonise_lab_units(labs: pd.DataFrame) -> pd.DataFrame:
    """Convert the mixed-unit analytes onto one unit each.

    The magnitude-based rules only run when the source unit is missing; a
    present-but-unknown unit is left untouched rather than guessed.
    """
    labs = labs.copy()
    if "unit" not in labs.columns:
        labs["unit"] = pd.NA

    lab_name = labs["lab_name"].astype("string").str.strip()
    unit = labs["unit"].astype("string").str.strip()

    factors = pd.Series(1.0, index=labs.index, dtype="float64")
    explicit = (
        ("Albumin", "g/L", 0.1),
        ("Creatinine", "umol/L", 1 / 88.4),
        ("Glucose", "mmol/L", 18.0182),
        ("Hemoglobin", "g/L", 0.1),
    )
    for name, source_unit, factor in explicit:
        factors.loc[lab_name.eq(name) & unit.eq(source_unit)] = factor

    missing = unit.isna()
    inferred = (
        ("Albumin", labs["value"].gt(10), 0.1),
        ("Creatinine", labs["value"].gt(20), 1 / 88.4),
        ("Glucose", labs["value"].le(30), 18.0182),
        ("Hemoglobin", labs["value"].gt(30), 0.1),
    )
    for name, condition, factor in inferred:
        factors.loc[missing & lab_name.eq(name) & condition] = factor

    labs["value"] = labs["value"] * factors
    return labs


def aggregate_labs(labs: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    require_columns(labs, "lab_results", [ID_COL, "lab_name", "value"])

    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs = _harmonise_lab_units(labs)

    if "abnormal_flag" in labs.columns:
        flags = labs["abnormal_flag"].astype("string").str.strip().str.upper()
        unknown = sorted(set(flags.dropna().unique()) - set(_ABNORMAL_MAP))
        if unknown:
            _log(
                f"  [warn] unrecognised abnormal_flag value(s) {unknown}; "
                "treated as missing.",
                verbose,
            )
        labs["_abnormal"] = flags.map(_ABNORMAL_MAP)
    else:
        labs["_abnormal"] = np.nan

    lab_values = labs.groupby([ID_COL, "lab_name"]).agg(
        count=("value", "count"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        std=("value", "std"),
    ).unstack("lab_name")
    lab_values.columns = [f"lab_{stat}_{name}" for stat, name in lab_values.columns]
    lab_values = lab_values.reset_index()

    lab_abnormal = labs.groupby([ID_COL, "lab_name"]).agg(
        abnormal_count=("_abnormal", "sum"),
        abnormal_rate=("_abnormal", "mean"),
    ).unstack("lab_name")
    lab_abnormal.columns = [f"lab_{stat}_{name}" for stat, name in lab_abnormal.columns]
    lab_abnormal = lab_abnormal.reset_index()

    lab_counts = labs.groupby(ID_COL).size().rename("total_labs").reset_index()
    return lab_counts.merge(lab_values, on=ID_COL, how="left").merge(
        lab_abnormal, on=ID_COL, how="left"
    )


def _harmonise_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    """Temperatures recorded in Fahrenheit are converted to Celsius."""
    vitals = vitals.copy()
    fahrenheit = (
        vitals["vital_type"].astype("string").str.upper().eq("TEMP")
        & vitals["value"].gt(50)
        & vitals["value"].le(120)
    )
    vitals.loc[fahrenheit, "value"] = (vitals.loc[fahrenheit, "value"] - 32) * (5 / 9)
    return vitals


def aggregate_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    require_columns(vitals, "vitals", [ID_COL, "vital_type", "value"])

    vitals["value"] = pd.to_numeric(vitals["value"], errors="coerce")
    vitals = _harmonise_vitals(vitals)

    aggregated = vitals.groupby([ID_COL, "vital_type"]).agg(
        count=("value", "count"),
        mean=("value", "mean"),
        min=("value", "min"),
        max=("value", "max"),
        std=("value", "std"),
    ).unstack("vital_type")
    aggregated.columns = [f"vital_{stat}_{kind}" for stat, kind in aggregated.columns]
    aggregated = aggregated.reset_index()

    counts = vitals.groupby(ID_COL).size().rename("total_vitals").reset_index()
    return counts.merge(aggregated, on=ID_COL, how="left")


def aggregate_transfers(transfers: pd.DataFrame) -> pd.DataFrame:
    require_columns(transfers, "unit_transfers", [ID_COL, "unit", "in_dt", "out_dt"])

    transfers["in_dt"] = _to_datetime(transfers["in_dt"])
    transfers["out_dt"] = _to_datetime(transfers["out_dt"])
    transfers["unit_hours"] = (
        transfers["out_dt"] - transfers["in_dt"]
    ).dt.total_seconds().div(3600).clip(lower=0)

    aggregated = transfers.groupby(ID_COL).agg(
        total_unit_transfers=("unit", "size"),
        total_unit_hours=("unit_hours", "sum"),
    ).reset_index()

    unit_hours = transfers.pivot_table(
        index=ID_COL,
        columns="unit",
        values="unit_hours",
        aggfunc="sum",
        fill_value=0,
    )
    unit_hours.columns = [f"unit_hours_{unit}" for unit in unit_hours.columns]
    unit_hours = unit_hours.reset_index()

    aggregated = aggregated.merge(unit_hours, on=ID_COL, how="left")
    return aggregated.merge(
        count_by_category(transfers, ID_COL, "unit", "unit_transfer_count"),
        on=ID_COL,
        how="left",
    )


def add_prior_features(master: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Prior utilisation, aggregated **per encounter** and cut off at admission.

    The previous implementation merged the prior-visit table on ``patient_id``,
    evaluated the 90-day window against every admission of that patient, and
    then summed with ``groupby("patient_id")``.  That inflated the count by the
    number of index encounters and, worse, credited encounter *A* with visits
    that happened before a *later* encounter *B* - i.e. after *A* was
    discharged.  Everything below is scoped to a single encounter.
    """
    require_columns(prior, "prior_utilization", [PATIENT_COL, "visit_dt", "visit_type"])

    prior = prior.copy()
    prior["visit_dt"] = _to_datetime(prior["visit_dt"])
    prior = prior.dropna(subset=["visit_dt", PATIENT_COL])
    prior[PATIENT_COL] = prior[PATIENT_COL].astype(master[PATIENT_COL].dtype)

    scoped = master[[ID_COL, PATIENT_COL, "admit_dt"]].merge(
        prior, on=PATIENT_COL, how="inner"
    )
    # Only visits strictly before this admission may describe this admission.
    scoped = scoped.loc[scoped["visit_dt"] < scoped["admit_dt"]].copy()
    scoped["_within_window"] = scoped["visit_dt"].ge(
        scoped["admit_dt"] - pd.Timedelta(days=PRIOR_WINDOW_DAYS)
    )

    per_encounter = scoped.groupby(ID_COL).agg(
        total_prior_visits=("visit_type", "size"),
        last_prior_visit_dt=("visit_dt", "max"),
        prior_visits_90d=("_within_window", "sum"),
    ).reset_index()

    master = master.merge(per_encounter, on=ID_COL, how="left")
    master = master.merge(
        count_by_category(scoped, ID_COL, "visit_type", "prior_visit_type_count"),
        on=ID_COL,
        how="left",
    )

    master["days_since_last_prior_visit"] = (
        master["admit_dt"] - master["last_prior_visit_dt"]
    ).dt.total_seconds() / 86400.0

    negative = master["days_since_last_prior_visit"].dropna() < 0
    if negative.any():
        raise ValueError(
            "prior_utilization contains a visit dated after the index admission; "
            "refusing to build a feature that looks into the future."
        )
    return master


def build_discharge_features(discharge: pd.DataFrame) -> pd.DataFrame:
    require_columns(discharge, "discharge_planning", [ID_COL])

    discharge = discharge.copy()
    if "followup_scheduled_days" in discharge.columns:
        discharge["followup_scheduled_days"] = pd.to_numeric(
            discharge["followup_scheduled_days"], errors="coerce"
        )
    for column in ("home_health_referral", "education_done"):
        if column in discharge.columns:
            discharge[column] = _to_binary(discharge[column])
    return discharge.drop_duplicates(subset=[ID_COL])


# --------------------------------------------------------------------------- #
# encoding + contract
# --------------------------------------------------------------------------- #
def _is_categorical_like(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def encode_categoricals(
    frame: pd.DataFrame,
    *,
    id_cols=ID_COLS,
    max_levels: int = MAX_CATEGORICAL_LEVELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-hot encode low-cardinality categoricals, drop the rest *visibly*."""
    frame = frame.copy()
    candidates = [
        column
        for column in frame.columns
        if column not in id_cols and _is_categorical_like(frame[column])
    ]

    to_encode: list[str] = []
    dropped: list[tuple[str, int]] = []
    for column in candidates:
        levels = int(frame[column].nunique(dropna=False))
        if levels <= max_levels:
            values = frame[column].astype("object")
            frame[column] = values.where(values.notna(), "Missing").astype(str)
            to_encode.append(column)
        else:
            dropped.append((column, levels))

    if dropped:
        _log(
            "  [encoding] dropped high-cardinality categorical(s): "
            + ", ".join(f"{name} ({levels} levels)" for name, levels in dropped),
            verbose,
        )
        frame = frame.drop(columns=[name for name, _ in dropped])
    if to_encode:
        frame = pd.get_dummies(frame, columns=to_encode, drop_first=True, dtype=np.uint8)
    return frame


def validate_features(
    frame: pd.DataFrame,
    *,
    name: str,
    require_target: bool = False,
    verbose: bool = True,
) -> None:
    """Assert the invariants every feature frame must satisfy."""
    problems: list[str] = []

    if ID_COL not in frame.columns:
        problems.append(f"missing {ID_COL!r}")
    else:
        if frame[ID_COL].isna().any():
            problems.append(f"null {ID_COL}")
        if frame[ID_COL].duplicated().any():
            problems.append(f"duplicate {ID_COL}")

    if require_target:
        if TARGET_COL not in frame.columns:
            problems.append(f"missing target {TARGET_COL!r}")
        else:
            labels = frame[TARGET_COL]
            if labels.isna().any():
                problems.append("null target values")
            elif not set(pd.unique(labels)) <= {0, 1}:
                problems.append("target is not binary 0/1")
    elif TARGET_COL in frame.columns:
        problems.append(f"target {TARGET_COL!r} present in a feature-only frame")

    leaked = leaky_columns(frame, ignore={TARGET_COL, *ID_COLS})
    if leaked:
        problems.append(f"outcome-like column(s) in the feature matrix: {leaked}")

    non_numeric = [
        column
        for column in frame.columns
        if column not in ID_COLS
        and not pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if non_numeric:
        problems.append(f"non-numeric feature column(s): {non_numeric[:5]}")

    numeric = frame.select_dtypes(include=[np.number])
    infinite = [
        column
        for column in numeric.columns
        if len(numeric[column])
        and np.isinf(numeric[column].dropna().to_numpy(dtype="float64")).any()
    ]
    if infinite:
        problems.append(f"infinite value(s) in: {infinite[:5]}")

    if problems:
        raise ValueError(f"{name}: feature validation failed:\n  - " + "\n  - ".join(problems))

    missing_share = (
        frame.drop(columns=[c for c in ID_COLS if c in frame.columns])
        .isna()
        .mean()
        .sort_values(ascending=False)
    )
    worst = missing_share.head(3)
    if len(worst) and worst.iloc[0] > 0:
        _log(
            "  [quality] most-missing features: "
            + ", ".join(f"{col} ({share:.0%})" for col, share in worst.items()),
            verbose,
        )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build_features(
    raw: dict[str, pd.DataFrame],
    *,
    strict_temporal: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build one feature row per index encounter, keyed by ``encounter_id``."""
    encounters = _read(raw, "index_encounters")
    patients = _read(raw, "patients")
    hospitals = _read(raw, "hospitals")

    _log("Creating encounter and patient features...", verbose)
    master = add_encounter_features(encounters, patients, hospitals)
    discharge_dt = master.set_index(ID_COL)["discharge_dt"]

    _log("Aggregating diagnoses...", verbose)
    diagnoses = _temporal_guard(
        _read(raw, "diagnoses"),
        "diagnoses",
        discharge_dt,
        _TIME_COLUMNS["diagnoses"],
        enabled=strict_temporal,
        verbose=verbose,
    )
    diag_agg = aggregate_diagnoses(diagnoses, _read(raw, "icd10_lookup"))

    _log("Aggregating procedures and medications...", verbose)
    procedures = _temporal_guard(
        _read(raw, "procedures"),
        "procedures",
        discharge_dt,
        _TIME_COLUMNS["procedures"],
        enabled=strict_temporal,
        verbose=verbose,
    )
    proc_agg = aggregate_procedures(procedures)

    medications = _temporal_guard(
        _read(raw, "medications"),
        "medications",
        discharge_dt,
        _TIME_COLUMNS["medications"],
        enabled=strict_temporal,
        verbose=verbose,
    )
    meds_agg = aggregate_medications(medications)

    _log("Aggregating labs and vitals by measurement type...", verbose)
    labs = _temporal_guard(
        _read(raw, "lab_results"),
        "lab_results",
        discharge_dt,
        _TIME_COLUMNS["lab_results"],
        enabled=strict_temporal,
        verbose=verbose,
    )
    labs_agg = aggregate_labs(labs, verbose=verbose)

    vitals = _temporal_guard(
        _read(raw, "vitals"),
        "vitals",
        discharge_dt,
        _TIME_COLUMNS["vitals"],
        enabled=strict_temporal,
        verbose=verbose,
    )
    vitals_agg = aggregate_vitals(vitals)

    _log("Aggregating unit transfers and prior utilization...", verbose)
    transfer_agg = aggregate_transfers(_read(raw, "unit_transfers"))

    _log("Loading discharge-planning features...", verbose)
    discharge_features = build_discharge_features(_read(raw, "discharge_planning"))

    _log("Merging features...", verbose)
    for features in (
        diag_agg,
        proc_agg,
        meds_agg,
        labs_agg,
        vitals_agg,
        transfer_agg,
        discharge_features,
    ):
        master = master.merge(features, on=ID_COL, how="left")
    master = add_prior_features(master, _read(raw, "prior_utilization"))

    master = master.drop(columns=list(_DROP_AFTER_BUILD), errors="ignore")

    zero_fill = [c for c in master.columns if c.startswith(_ZERO_FILL_PREFIXES)]
    zero_fill += [c for c in _ZERO_FILL_COLUMNS if c in master.columns and c not in zero_fill]
    if zero_fill:
        master[zero_fill] = master[zero_fill].fillna(0.0)

    _log("Encoding categorical features...", verbose)
    master = encode_categoricals(master, verbose=verbose)

    numeric = master.select_dtypes(include=[np.number]).columns
    if len(numeric):
        master[numeric] = master[numeric].replace([np.inf, -np.inf], np.nan)

    validate_features(master, name="features", require_target=False, verbose=verbose)
    return master


def split_and_write(
    master: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cleaned_dir: str = CLEANED_DIR,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach labels to train rows, ids to test rows, and write both files."""
    require_columns(train_df, "train", [ID_COL, TARGET_COL])
    require_columns(test_df, "test", [ID_COL])

    if TARGET_COL in master.columns:
        raise ValueError(
            f"{TARGET_COL!r} is already a column of the feature frame; "
            "the index-encounter table must not carry the label."
        )

    final_train = train_df[[ID_COL, TARGET_COL]].merge(
        master, on=ID_COL, how="inner", validate="one_to_one"
    )
    final_test = test_df[[ID_COL]].merge(
        master, on=ID_COL, how="inner", validate="one_to_one"
    )

    if len(final_train) != len(train_df):
        raise ValueError(
            f"lost {len(train_df) - len(final_train)} train encounter(s) during the feature merge"
        )
    if len(final_test) != len(test_df):
        raise ValueError(
            f"lost {len(test_df) - len(final_test)} test encounter(s) during the feature merge"
        )

    train_features = [c for c in final_train.columns if c != TARGET_COL]
    if train_features != list(final_test.columns):
        only_train = [c for c in train_features if c not in final_test.columns]
        only_test = [c for c in final_test.columns if c not in train_features]
        raise ValueError(
            "train/test feature columns differ after the split; "
            f"train-only={only_train[:5]} test-only={only_test[:5]}"
        )

    validate_features(final_train, name="train", require_target=True, verbose=verbose)
    validate_features(final_test, name="test", require_target=False, verbose=verbose)

    os.makedirs(cleaned_dir, exist_ok=True)
    final_train.to_csv(os.path.join(cleaned_dir, "train_processed.csv"), index=False)
    final_test.to_csv(os.path.join(cleaned_dir, "test_processed.csv"), index=False)

    _log(
        f"Pipeline complete! Train shape: {final_train.shape} | Test shape: {final_test.shape}",
        verbose,
    )
    return final_train, final_test


def main() -> None:
    print(f"Raw dir     : {RAW_DIR}")
    print(f"Cleaned dir : {CLEANED_DIR}")
    raw = load_raw(RAW_DIR)
    master = build_features(raw)
    split_and_write(master, raw["train"], raw["test"], CLEANED_DIR)


if __name__ == "__main__":
    main()
