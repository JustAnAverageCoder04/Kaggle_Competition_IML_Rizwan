"""Single source of truth for the data contract shared by every pipeline stage.

Both the feature builder (``garbage_duty``) and the modelling helpers
(``modeling_utils``) import these constants, so ``encounter_id`` cannot be
spelled differently in one script than in another, and the "no outcome column
may become a feature" rule is defined exactly once.
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "readmit_30d"
ID_COL = "encounter_id"
PATIENT_COL = "patient_id"
HOSPITAL_COL = "hospital_id"

#: Identifiers. Kept in the cleaned files for grouping/auditing, never modelled.
ID_COLS = (ID_COL, PATIENT_COL, HOSPITAL_COL)

#: Substrings that mark a column as an outcome. Matching columns abort the run.
FORBIDDEN_TOKENS = (
    "readmit",
    "target",
    "label",
    "outcome",
    "death",
    "died",
    "expire",
    "deceased",
    "risk_score",
)

#: Substrings for fields that look risky but are legitimately known at
#: discharge and are therefore allowed through (e.g. ``discharge_disposition``).
EXEMPT_TOKENS = ("disposition",)


def leaky_columns(frame: pd.DataFrame, *, ignore=()) -> list[str]:
    """Return columns whose *names* suggest an outcome leaking into the features."""
    ignore = set(ignore)
    found: list[str] = []
    for column in frame.columns:
        if column in ignore:
            continue
        name = str(column).lower()
        if any(token in name for token in EXEMPT_TOKENS):
            continue
        if any(token in name for token in FORBIDDEN_TOKENS):
            found.append(column)
    return found


def require_columns(frame: pd.DataFrame, table: str, columns) -> None:
    """Fail loudly when a source table does not have the columns we rely on."""
    missing = [c for c in columns if c not in frame.columns]
    if not missing:
        return
    preview = ", ".join(map(str, list(frame.columns)[:15])) or "<no columns>"
    raise KeyError(
        f"{table}: missing required column(s) {missing}. "
        f"Columns present (first 15): {preview}"
    )


def check_binary_target(series: pd.Series, name: str = TARGET_COL) -> pd.Series:
    """Validate a 0/1 target column and return it as ``int8``."""
    if series.isna().any():
        raise ValueError(f"{name}: {int(series.isna().sum())} missing label(s)")
    values = set(pd.unique(series))
    if not values <= {0, 1}:
        raise ValueError(f"{name}: expected a 0/1 label, found {sorted(values)[:10]}")
    return series.astype("int8")
