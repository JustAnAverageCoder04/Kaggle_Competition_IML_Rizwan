import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading processed datasets...")
train_df = pd.read_csv(os.path.join(CLEANED_DIR, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(CLEANED_DIR, "test_processed.csv"))

target_col = "readmit_30d"
id_col = "encounter_id"

# 1. Extract groups (hospital_id) before dropping ID/target columns
groups = train_df["hospital_id"] if "hospital_id" in train_df.columns else None

# 2. Drop non-feature columns from training data
X = train_df.drop(columns=[id_col, target_col, "hospital_id"], errors="ignore")
y = train_df[target_col]

X_test = test_df.drop(columns=[id_col, "hospital_id"], errors="ignore")
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Filter to numeric/boolean types and handle NaNs
X = X.select_dtypes(include=[np.number, "bool"]).fillna(-999)
X_test = X_test.select_dtypes(include=[np.number, "bool"]).fillna(-999)

print(f"Training with StratifiedGroupKFold on {X.shape[1]} features.")

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Use your best tuned parameters from Optuna here
model = LGBMClassifier(
  n_estimators=362,
  learning_rate=0.024392978571193134,
  max_depth=7,
  num_leaves=15,
  min_child_samples=74,
  scale_pos_weight=9,
  subsample_freq = 1,
  subsample=0.6054290714457558,
  colsample_bytree=0.6673719904752734,
  reg_alpha=0.0012492195382690268,
  reg_lambda=0.020730321132301593
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Stratified Group Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} Grouped ROC-AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Grouped Out-of-Fold ROC-AUC: {overall_auc:.4f}")

# Save OOF predictions and submission file
np.save(os.path.join(MODELS_DIR, "oof_preds_lgb.npy"), oof_preds)
submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_preds
})
submission.to_csv(os.path.join(MODELS_DIR, "submission_lgb.csv"), index=False)
print("Saved updated grouped LightGBM submission.")