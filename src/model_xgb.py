import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Dynamic path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading processed datasets for XGBoost...")
train_df = pd.read_csv(os.path.join(CLEANED_DIR, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(CLEANED_DIR, "test_processed.csv"))

target_col = "readmit_30d"
id_col = "encounter_id"

groups = train_df["hospital_id"]

X = train_df.drop(columns=[id_col, target_col, "hospital_id"], errors="ignore")
y = train_df[target_col]
X_test = test_df.drop(columns=[id_col, "hospital_id"], errors="ignore")

# Align test columns with train columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Filter strictly to numeric/boolean types
X = X.select_dtypes(include=[np.number, "bool"])
X_test = X_test.select_dtypes(include=[np.number, "bool"])

# Handle NaNs
#X = X.fillna(-999)
#X_test = X_test.fillna(-999)

print(f"Training XGBoost on {X.shape[1]} clean numeric features.")

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

# XGBoost Classifier setup with histogram-based splitting for speed
model = XGBClassifier(
  n_estimators=742,
  learning_rate=0.021741650139434836,
  max_depth=3,
  min_child_weight=8,
  subsample=0.8501817598839572,
  colsample_bytree=0.6752675542666599,
  reg_alpha=0.01686207755998821,
  reg_lambda=0.4054985340213573
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Stratified Group Cross-Validation with XGBoost...")
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} Grouped XGBoost ROC-AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Grouped Out-of-Fold XGBoost ROC-AUC: {overall_auc:.4f}")

# Save Submission
submission = pd.DataFrame({
    "encounter_id": test_df["encounter_id"],
    "readmit_30d": test_preds
})

sub_path = os.path.join(MODELS_DIR, "submission_xgb.csv")
submission.to_csv(sub_path, index=False)
print(f"XGBoost submission saved to: models/submission_xgb.csv")