import os
import pandas as pd
import numpy as np

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

# Frequency encode high-cardinality attending_id instead of dropping it
attending_freq = encounters_df["attending_id"].value_counts(normalize=True)
encounters_df["attending_freq_encoded"] = encounters_df["attending_id"].map(attending_freq)

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
    unique_procedure_codes=("code", "nunique"),
).reset_index()
proc_agg = proc_agg.merge(
    count_by_category(procedures_df, "encounter_id", "code_type", "procedure_type_count"),
    on="encounter_id",
    how="left",
)

meds_df = pd.read_csv(os.path.join(RAW_DIR, "medications.csv"))
meds_agg = meds_df.groupby("encounter_id").agg(
    unique_drug_classes=("drug_class", "nunique"),
    medications_continued_at_discharge=("continued_at_discharge", "sum"),
).reset_index()
for category, prefix in [("drug_class", "medication_class_count"), ("route", "medication_route_count")]:
    meds_agg = meds_agg.merge(
        count_by_category(meds_df, "encounter_id", category, prefix),
        on="encounter_id",
        how="left",
    )

print("Aggregating labs and vitals (streamlined)...")
labs_df = pd.read_csv(os.path.join(RAW_DIR, "lab_results.csv"))
labs_df["value"] = pd.to_numeric(labs_df["value"], errors="coerce")

# Streamlined lab value aggregation (mean and max are usually sufficient for tree models)
lab_values = labs_df.groupby(["encounter_id", "lab_name"]).agg(
    mean=("value", "mean"),
    max=("value", "max"),
).unstack("lab_name")
lab_values.columns = [f"lab_{stat}_{name}" for stat, name in lab_values.columns]
lab_values = lab_values.reset_index()

lab_abnormal = labs_df.assign(
    _abnormal=labs_df["abnormal_flag"].map({"H": 1.0, "L": 1.0, "N": 0.0})
).groupby(["encounter_id", "lab_name"]).agg(
    abnormal_rate=("_abnormal", "mean"),
).unstack("lab_name")
lab_abnormal.columns = [f"lab_abnormal_rate_{name}" for stat, name in lab_abnormal.columns]
lab_abnormal = lab_abnormal.reset_index()

lab_counts = labs_df.groupby("encounter_id").size().rename("total_labs").reset_index()
labs_agg = lab_counts.merge(lab_values, on="encounter_id", how="left").merge(
    lab_abnormal, on="encounter_id", how="left"
)

vitals_df = pd.read_csv(os.path.join(RAW_DIR, "vitals.csv"))
vitals_df["value"] = pd.to_numeric(vitals_df["value"], errors="coerce")
vitals_agg = vitals_df.groupby(["encounter_id", "vital_type"]).agg(
    mean=("value", "mean"),
    max=("value", "max"),
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

prior_df = pd.read_csv(os.path.join(RAW_DIR, "prior_utilization.csv"))
prior_df["visit_dt"] = pd.to_datetime(prior_df["visit_dt"], errors="coerce")
prior_agg = prior_df.groupby("patient_id").agg(
    total_prior_visits=("visit_type", "size"),
    last_prior_visit_dt=("visit_dt", "max"),
).reset_index()

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
discharge_df["home_health_referral"] = discharge_df["home_health_referral"].map({"Y": 1, "N": 0})
discharge_df["education_done"] = discharge_df["education_done"].map({"Yes": 1, "No": 0})
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
    master_df[col] = master_df[col].fillna("Missing").astype(str)

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

final_train.to_csv(os.path.join(CLEANED_DIR, "train_processed.csv"), index=False)
final_test.to_csv(os.path.join(CLEANED_DIR, "test_processed.csv"), index=False)
print(f"Pipeline complete! Train shape: {final_train.shape} | Test shape: {final_test.shape}")