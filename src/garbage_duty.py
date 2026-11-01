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

# 1. Explicitly drop text/name/timestamp/description columns by name to prevent bloat
print("Dropping non-informative schema text/timestamp columns...")
patients_df = patients_df.drop(columns=[col for col in patients_df.columns if any(w in col.lower() for w in ["name", "date", "time", "address", "phone"])], errors="ignore")
hospitals_df = hospitals_df.drop(columns=[col for col in hospitals_df.columns if any(w in col.lower() for w in ["name", "address", "city", "state"])], errors="ignore")
encounters_df = encounters_df.drop(columns=[col for col in encounters_df.columns if any(w in col.lower() for w in ["date", "time", "timestamp", "desc", "text"])], errors="ignore")

# 2. Aggregate Child Tables strictly to counts/metrics (No raw text strings brought in)
print("Aggregating multi-row child tables...")

diagnoses_df = pd.read_csv(os.path.join(RAW_DIR, "diagnoses.csv"))
diag_agg = diagnoses_df.groupby("encounter_id").size().reset_index(name="total_diagnoses")

procedures_df = pd.read_csv(os.path.join(RAW_DIR, "procedures.csv"))
proc_agg = procedures_df.groupby("encounter_id").size().reset_index(name="total_procedures")

meds_df = pd.read_csv(os.path.join(RAW_DIR, "medications.csv"))
meds_agg = meds_df.groupby("encounter_id").size().reset_index(name="total_medications")

labs_df = pd.read_csv(os.path.join(RAW_DIR, "lab_results.csv"))
labs_agg = labs_df.groupby("encounter_id").size().reset_index(name="total_lab_tests")

vitals_df = pd.read_csv(os.path.join(RAW_DIR, "vitals.csv"))
vitals_agg = vitals_df.groupby("encounter_id").size().reset_index(name="total_vitals_recorded")

transfers_df = pd.read_csv(os.path.join(RAW_DIR, "unit_transfers.csv"))
transfers_agg = transfers_df.groupby("encounter_id").size().reset_index(name="total_unit_transfers")

# 3. Aggregate Prior Utilization by Patient Grain
print("Aggregating prior utilization...")
prior_df = pd.read_csv(os.path.join(RAW_DIR, "prior_utilization.csv"))
prior_agg = prior_df.groupby("patient_id").size().reset_index(name="total_prior_visits")

# 4. Merge All Tables onto Index Encounters
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

# 5. Handle Missing Values in Counts
count_cols = [
    "total_diagnoses", "total_procedures", "total_medications", 
    "total_lab_tests", "total_vitals_recorded", "total_unit_transfers", "total_prior_visits"
]
master_df[count_cols] = master_df[count_cols].fillna(0)

# 6. Safe, minimal categorical encoding (Only encode columns with <= 10 unique values)
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
        # Drop any remaining high-cardinality text columns that slipped through
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