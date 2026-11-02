import os
import pandas as pd
import numpy as np

# Get the directory where this script is located (src/), then go up one level to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)

print("Loading base files...")
train_df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
encounters_df = pd.read_csv(os.path.join(RAW_DIR, "index_encounters.csv"))
patients_df = pd.read_csv(os.path.join(RAW_DIR, "patients.csv"))
hospitals_df = pd.read_csv(os.path.join(RAW_DIR, "hospitals.csv"))

# 1. Extract Length of Stay (LOS) BEFORE dropping timestamps
print("Calculating Length of Stay...")
if "admit_dt" in encounters_df.columns and "discharge_dt" in encounters_df.columns:
    encounters_df["admission_date"] = pd.to_datetime(encounters_df["admit_dt"], errors="coerce")
    encounters_df["discharge_date"] = pd.to_datetime(encounters_df["discharge_dt"], errors="coerce")
    encounters_df["length_of_stay_days"] = (encounters_df["discharge_date"] - encounters_df["admission_date"]).dt.total_seconds() / 86400.0
else:
    encounters_df["length_of_stay_days"] = 0.0

# 2. Compute Hospital Patient Volume / Institutional Load
print("Computing hospital-level features...")
hospital_vol = encounters_df.groupby("hospital_id").size().reset_index(name="hospital_encounter_volume")
hospitals_df = hospitals_df.merge(hospital_vol, on="hospital_id", how="left")

# Drop remaining non-informative text/timestamp columns safely
patients_df = patients_df.drop(columns=[col for col in patients_df.columns if any(w in col.lower() for w in ["name", "date", "time", "address", "phone"])], errors="ignore")
hospitals_df = hospitals_df.drop(columns=[col for col in hospitals_df.columns if any(w in col.lower() for w in ["name", "address", "city", "state"])], errors="ignore")
encounters_df = encounters_df.drop(columns=[col for col in encounters_df.columns if any(w in col.lower() for w in ["date", "time", "timestamp", "desc", "text"])], errors="ignore")

# 3. Aggregate Child Tables with Statistical Summaries & Missing Flags
print("Aggregating multi-row child tables and building missingness flags...")

diagnoses_df = pd.read_csv(os.path.join(RAW_DIR, "diagnoses.csv"))
diag_agg = diagnoses_df.groupby("encounter_id").agg(
    total_diagnoses=("icd10_code", "count"),
    unique_diagnoses=("icd10_code", "nunique")
).reset_index()

procedures_df = pd.read_csv(os.path.join(RAW_DIR, "procedures.csv"))
proc_agg = procedures_df.groupby("encounter_id").size().reset_index(name="total_procedures")

meds_df = pd.read_csv(os.path.join(RAW_DIR, "medications.csv"))
meds_agg = meds_df.groupby("encounter_id").size().reset_index(name="total_medications")

labs_df = pd.read_csv(os.path.join(RAW_DIR, "lab_results.csv"))
if "result_value" in labs_df.columns:
    labs_agg = labs_df.groupby("encounter_id").agg(
        total_labs=("result_value", "count"),
        mean_lab_value=("result_value", "mean"),
        max_lab_value=("result_value", "max"),
        min_lab_value=("result_value", "min")
    ).reset_index()
else:
    labs_agg = labs_df.groupby("encounter_id").size().reset_index(name="total_labs")

vitals_df = pd.read_csv(os.path.join(RAW_DIR, "vitals.csv"))
if "vital_value" in vitals_df.columns:
    vitals_agg = vitals_df.groupby("encounter_id").agg(
        total_vitals=("vital_value", "count"),
        mean_vital=("vital_value", "mean"),
        max_vital=("vital_value", "max"),
        min_vital=("vital_value", "min")
    ).reset_index()
else:
    vitals_agg = vitals_df.groupby("encounter_id").size().reset_index(name="total_vitals")

transfers_df = pd.read_csv(os.path.join(RAW_DIR, "unit_transfers.csv"))
transfers_agg = transfers_df.groupby("encounter_id").size().reset_index(name="total_unit_transfers")

prior_df = pd.read_csv(os.path.join(RAW_DIR, "prior_utilization.csv"))
prior_agg = prior_df.groupby("patient_id").agg(
    total_prior_visits=("patient_id", "count")
).reset_index()

# 4. Merge All Tables into Master Table
print("Merging tables into master table...")
master_df = encounters_df.merge(patients_df, on="patient_id", how="left")
master_df = master_df.merge(hospitals_df, on="hospital_id", how="left")
master_df = master_df.merge(diag_agg, on="encounter_id", how="left")
master_df = master_df.merge(proc_agg, on="encounter_id", how="left")
master_df = master_df.merge(meds_agg, on="encounter_id", how="left")
master_df = master_df.merge(labs_agg, on="encounter_id", how="left")
master_df = master_df.merge(vitals_agg, on="encounter_id", how="left")
master_df = master_df.merge(transfers_agg, on="encounter_id", how="left")
master_df = master_df.merge(prior_agg, on="patient_id", how="left")

# 5. Fill Missing Values & Create Explicit Missingness Indicator Flags
numeric_aggregates = [
    "total_diagnoses", "unique_diagnoses", "total_procedures", "total_medications", 
    "total_labs", "mean_lab_value", "max_lab_value", "min_lab_value",
    "total_vitals", "mean_vital", "max_vital", "min_vital", 
    "total_unit_transfers", "total_prior_visits", "length_of_stay_days"
]
existing_num_cols = [col for col in numeric_aggregates if col in master_df.columns]

for col in existing_num_cols:
    # Create binary indicator for whether data was missing before imputation
    master_df[f"{col}_was_missing"] = master_df[col].isna().astype(int)
    master_df[col] = master_df[col].fillna(0)

# 6. Safe Categorical Encoding (<= 10 Unique Values)
print("Encoding low-cardinality categoricals...")
categorical_cols = master_df.select_dtypes(include=["object", "category"]).columns.tolist()

low_card_cats = []
for col in categorical_cols:
    if col in ["encounter_id", "patient_id", "hospital_id"]:
        continue
    if master_df[col].nunique(dropna=False) <= 10:
        low_card_cats.append(col)
        master_df[col] = master_df[col].fillna("Missing").astype(str)
    else:
        master_df = master_df.drop(columns=[col])

if low_card_cats:
    master_df = pd.get_dummies(master_df, columns=low_card_cats, drop_first=True)

# 7. Split Back into Train and Test Sets
print("Splitting into final train and test sets...")
final_train = train_df[["encounter_id", "readmit_30d"]].merge(master_df, on="encounter_id", how="inner")
final_test = test_df[["encounter_id"]].merge(master_df, on="encounter_id", how="inner")

# 8. Export to Output Directory
final_train.to_csv(os.path.join(CLEANED_DIR, "train_processed.csv"), index=False)
final_test.to_csv(os.path.join(CLEANED_DIR, "test_processed.csv"), index=False)

print(f"Pipeline complete! Output files saved to {CLEANED_DIR}/")
print(f"Train shape: {final_train.shape} | Test shape: {final_test.shape}")