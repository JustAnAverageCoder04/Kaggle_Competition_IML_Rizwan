import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)


def count_by_category(df, key, category, prefix):
    values = df[[key, category]].dropna(subset=[category]).copy()
    values[category] = values[category].astype(str)
    counts = values.groupby([key, category]).size().unstack(fill_value=0)
    counts.columns = [f"{prefix}_{value}" for value in counts.columns]
    return counts.reset_index()


print("Loading base files...")
train_df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
encounters_df = pd.read_csv(os.path.join(RAW_DIR, "index_encounters.csv"))
patients_df = pd.read_csv(os.path.join(RAW_DIR, "patients.csv"))
hospitals_df = pd.read_csv(os.path.join(RAW_DIR, "hospitals.csv"))

print("Creating encounter and patient features...")
encounters_df["admit_dt"] = pd.to_datetime(encounters_df["admit_dt"], errors="coerce")
encounters_df["discharge_dt"] = pd.to_datetime(encounters_df["discharge_dt"], errors="coerce")
encounters_df["length_of_stay_days"] = (
    encounters_df["discharge_dt"] - encounters_df["admit_dt"]
).dt.total_seconds() / 86400.0
encounters_df["admission_month"] = encounters_df["admit_dt"].dt.month
encounters_df["admission_dayofweek"] = encounters_df["admit_dt"].dt.dayofweek
encounters_df["admission_hour"] = encounters_df["admit_dt"].dt.hour

patients_df["birth_year"] = pd.to_numeric(patients_df["birth_year"], errors="coerce")
birth_years = patients_df.set_index("patient_id")["birth_year"]
encounters_df["age_at_admission"] = (
    encounters_df["admit_dt"].dt.year - encounters_df["patient_id"].map(birth_years)
)

hospital_vol = encounters_df.groupby("hospital_id").size().rename("hospital_encounter_volume")
hospitals_df = hospitals_df.merge(hospital_vol, on="hospital_id", how="left")

print("Aggregating diagnoses...")
diagnoses_df = pd.read_csv(os.path.join(RAW_DIR, "diagnoses.csv"))
lookup_df = pd.read_csv(os.path.join(RAW_DIR, "icd10_lookup.csv"))
diagnoses_df["_code_key"] = (
    diagnoses_df["icd10_code"].astype("string").str.replace(".", "", regex=False).str.upper()
)
lookup_df["_code_key"] = (
    lookup_df["icd10_code"].astype("string").str.replace(".", "", regex=False).str.upper()
)
diagnoses_df = diagnoses_df.merge(
    lookup_df[["_code_key", "chapter", "chronic_flag"]].drop_duplicates("_code_key"),
    on="_code_key",
    how="left",
)

diag_agg = diagnoses_df.groupby("encounter_id").agg(
    total_diagnoses=("icd10_code", "count"),
    unique_diagnoses=("icd10_code", "nunique"),
    primary_diagnoses=("is_primary", "sum"),
    chronic_diagnoses=("chronic_flag", "sum"),
    poa_yes=("present_on_admission", lambda values: values.eq("Y").sum()),
    poa_no=("present_on_admission", lambda values: values.eq("N").sum()),
).reset_index()
diag_agg = diag_agg.merge(
    count_by_category(diagnoses_df, "encounter_id", "chapter", "diagnosis_chapter_count"),
    on="encounter_id",
    how="left",
)

print("Aggregating procedures and medications...")
procedures_df = pd.read_csv(os.path.join(RAW_DIR, "procedures.csv"))
proc_agg = procedures_df.groupby("encounter_id").agg(
    total_procedures=("code", "size"),
    unique_procedure_codes=("code", "nunique"),
).reset_index()
proc_agg = proc_agg.merge(
    count_by_category(procedures_df, "encounter_id", "code", "procedure_code_count"),
    on="encounter_id",
    how="left",
)
proc_agg = proc_agg.merge(
    count_by_category(procedures_df, "encounter_id", "code_type", "procedure_type_count"),
    on="encounter_id",
    how="left",
)

meds_df = pd.read_csv(os.path.join(RAW_DIR, "medications.csv"))
meds_agg = meds_df.groupby("encounter_id").agg(
    total_medications=("drug_class", "size"),
    unique_drug_classes=("drug_class", "nunique"),
    medications_continued_at_discharge=("continued_at_discharge", "sum"),
).reset_index()
for category, prefix in [("drug_class", "medication_class_count"), ("route", "medication_route_count")]:
    meds_agg = meds_agg.merge(
        count_by_category(meds_df, "encounter_id", category, prefix),
        on="encounter_id",
        how="left",
    )

print("Aggregating labs and vitals by measurement type...")
labs_df = pd.read_csv(os.path.join(RAW_DIR, "lab_results.csv"))
labs_df["value"] = pd.to_numeric(labs_df["value"], errors="coerce")

# Convert mixed-unit lab results to a consistent unit per analyte.
lab_unit_factors = pd.Series(1.0, index=labs_df.index)
lab_unit_factors.loc[
    labs_df["lab_name"].eq("Albumin") & labs_df["unit"].eq("g/L")
] = 0.1
lab_unit_factors.loc[
    labs_df["lab_name"].eq("Creatinine") & labs_df["unit"].eq("umol/L")
] = 1 / 88.4
lab_unit_factors.loc[
    labs_df["lab_name"].eq("Glucose") & labs_df["unit"].eq("mmol/L")
] = 18.0182
lab_unit_factors.loc[
    labs_df["lab_name"].eq("Hemoglobin") & labs_df["unit"].eq("g/L")
] = 0.1

# Infer the common alternate unit only when the source unit is missing.
missing_unit = labs_df["unit"].isna()
lab_unit_factors.loc[
    missing_unit & labs_df["lab_name"].eq("Albumin") & labs_df["value"].gt(10)
] = 0.1
lab_unit_factors.loc[
    missing_unit & labs_df["lab_name"].eq("Creatinine") & labs_df["value"].gt(20)
] = 1 / 88.4
lab_unit_factors.loc[
    missing_unit & labs_df["lab_name"].eq("Glucose") & labs_df["value"].le(30)
] = 18.0182
lab_unit_factors.loc[
    missing_unit & labs_df["lab_name"].eq("Hemoglobin") & labs_df["value"].gt(30)
] = 0.1
labs_df["value"] *= lab_unit_factors

lab_values = labs_df.groupby(["encounter_id", "lab_name"]).agg(
    count=("value", "count"),
    mean=("value", "mean"),
    min=("value", "min"),
    max=("value", "max"),
    std=("value", "std"),
).unstack("lab_name")
lab_values.columns = [f"lab_{stat}_{name}" for stat, name in lab_values.columns]
lab_values = lab_values.reset_index()
lab_abnormal = labs_df.assign(
    _abnormal=labs_df["abnormal_flag"].map({"H": 1.0, "L": 1.0, "N": 0.0})
).groupby(["encounter_id", "lab_name"]).agg(
    abnormal_count=("_abnormal", "sum"),
    abnormal_rate=("_abnormal", "mean"),
).unstack("lab_name")
lab_abnormal.columns = [f"lab_{stat}_{name}" for stat, name in lab_abnormal.columns]
lab_abnormal = lab_abnormal.reset_index()
lab_counts = labs_df.groupby("encounter_id").size().rename("total_labs").reset_index()
labs_agg = lab_counts.merge(lab_values, on="encounter_id", how="left").merge(
    lab_abnormal, on="encounter_id", how="left"
)

vitals_df = pd.read_csv(os.path.join(RAW_DIR, "vitals.csv"))
vitals_df["value"] = pd.to_numeric(vitals_df["value"], errors="coerce")
temperature_fahrenheit = (
    vitals_df["vital_type"].eq("TEMP")
    & vitals_df["value"].gt(50)
    & vitals_df["value"].le(120)
)
vitals_df.loc[temperature_fahrenheit, "value"] = (
    vitals_df.loc[temperature_fahrenheit, "value"] - 32
) * (5 / 9)

vitals_agg = vitals_df.groupby(["encounter_id", "vital_type"]).agg(
    count=("value", "count"),
    mean=("value", "mean"),
    min=("value", "min"),
    max=("value", "max"),
    std=("value", "std"),
).unstack("vital_type")
vitals_agg.columns = [f"vital_{stat}_{kind}" for stat, kind in vitals_agg.columns]
vitals_agg = vitals_agg.reset_index()
vital_counts = vitals_df.groupby("encounter_id").size().rename("total_vitals").reset_index()
vitals_agg = vital_counts.merge(vitals_agg, on="encounter_id", how="left")

print("Aggregating unit transfers and prior utilization...")
transfers_df = pd.read_csv(os.path.join(RAW_DIR, "unit_transfers.csv"))
transfers_df["in_dt"] = pd.to_datetime(transfers_df["in_dt"], errors="coerce")
transfers_df["out_dt"] = pd.to_datetime(transfers_df["out_dt"], errors="coerce")
transfers_df["unit_hours"] = (
    transfers_df["out_dt"] - transfers_df["in_dt"]
).dt.total_seconds().div(3600).clip(lower=0)
transfer_agg = transfers_df.groupby("encounter_id").agg(
    total_unit_transfers=("unit", "size"),
    total_unit_hours=("unit_hours", "sum"),
).reset_index()
unit_hours = transfers_df.pivot_table(
    index="encounter_id",
    columns="unit",
    values="unit_hours",
    aggfunc="sum",
    fill_value=0,
)
unit_hours.columns = [f"unit_hours_{unit}" for unit in unit_hours.columns]
unit_hours = unit_hours.reset_index()
transfer_agg = transfer_agg.merge(unit_hours, on="encounter_id", how="left").merge(
    count_by_category(transfers_df, "encounter_id", "unit", "unit_transfer_count"),
    on="encounter_id",
    how="left",
)

prior_df = pd.read_csv(os.path.join(RAW_DIR, "prior_utilization.csv"))
prior_df["visit_dt"] = pd.to_datetime(prior_df["visit_dt"], errors="coerce")
prior_agg = prior_df.groupby("patient_id").agg(
    total_prior_visits=("visit_type", "size"),
    last_prior_visit_dt=("visit_dt", "max"),
).reset_index()
prior_agg = prior_agg.merge(
    count_by_category(prior_df, "patient_id", "visit_type", "prior_visit_type_count"),
    on="patient_id",
    how="left",
)
prior_at_index = prior_df.merge(
    encounters_df[["patient_id", "admit_dt"]], on="patient_id", how="inner"
)
prior_at_index["_recent_90"] = (
    prior_at_index["visit_dt"].ge(prior_at_index["admit_dt"] - pd.Timedelta(days=90))
    & prior_at_index["visit_dt"].le(prior_at_index["admit_dt"])
)
prior_90d = prior_at_index.groupby("patient_id")["_recent_90"].sum().rename(
    "prior_visits_90d"
).reset_index()
prior_agg = prior_agg.merge(prior_90d, on="patient_id", how="left")

print("Loading discharge-planning features...")
discharge_df = pd.read_csv(os.path.join(RAW_DIR, "discharge_planning.csv"))
discharge_df["followup_scheduled_days"] = pd.to_numeric(
    discharge_df["followup_scheduled_days"], errors="coerce"
)
discharge_df["home_health_referral"] = discharge_df["home_health_referral"].map(
    {"Y": 1, "N": 0}
)
discharge_df["education_done"] = discharge_df["education_done"].map(
    {"Yes": 1, "No": 0}
)
discharge_features = discharge_df

print("Merging features...")
master_df = encounters_df.merge(patients_df, on="patient_id", how="left", suffixes=("", "_patient"))
master_df = master_df.merge(hospitals_df, on="hospital_id", how="left")
for features in [
    diag_agg,
    proc_agg,
    meds_agg,
    labs_agg,
    vitals_agg,
    transfer_agg,
    discharge_features,
]:
    master_df = master_df.merge(features, on="encounter_id", how="left")
master_df = master_df.merge(prior_agg, on="patient_id", how="left")
master_df["days_since_last_prior_visit"] = (
    master_df["admit_dt"] - master_df["last_prior_visit_dt"]
).dt.total_seconds().div(86400)

count_prefixes = (
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
    "prior_visit_type_count_",
)
count_features = [
    col for col in master_df.columns if col.startswith(count_prefixes)
]
master_df[count_features] = master_df[count_features].fillna(0)
master_df["prior_visits_90d"] = master_df["prior_visits_90d"].fillna(0)
master_df["medications_continued_at_discharge"] = master_df[
    "medications_continued_at_discharge"
].fillna(0)

master_df = master_df.drop(
    columns=["admit_dt", "discharge_dt", "last_prior_visit_dt", "attending_id"],
    errors="ignore",
)

print("Encoding categorical features...")
categorical_cols = master_df.select_dtypes(include=["object", "category"]).columns.tolist()
excluded_ids = {"encounter_id", "patient_id", "hospital_id"}
for col in categorical_cols:
    if col in excluded_ids:
        continue
    if master_df[col].nunique(dropna=False) <= 25:
        master_df[col] = master_df[col].fillna("Missing").astype(str)
    else:
        master_df = master_df.drop(columns=[col])
categorical_cols = [
    col for col in master_df.select_dtypes(include=["object", "category"]).columns
    if col not in excluded_ids
]
if categorical_cols:
    master_df = pd.get_dummies(master_df, columns=categorical_cols, drop_first=True, dtype=np.uint8)

master_df = master_df.replace([np.inf, -np.inf], np.nan)

print("Writing processed train and test files...")
final_train = train_df[["encounter_id", "readmit_30d"]].merge(
    master_df, on="encounter_id", how="inner", validate="one_to_one"
)
final_test = test_df[["encounter_id"]].merge(
    master_df, on="encounter_id", how="inner", validate="one_to_one"
)
if len(final_train) != len(train_df) or len(final_test) != len(test_df):
    raise ValueError("Some train or test encounters were lost during feature merges.")

final_train.to_csv(os.path.join(CLEANED_DIR, "train_processed.csv"), index=False)
final_test.to_csv(os.path.join(CLEANED_DIR, "test_processed.csv"), index=False)

print(f"Pipeline complete! Train shape: {final_train.shape} | Test shape: {final_test.shape}")
