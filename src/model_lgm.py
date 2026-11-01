import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Dynamic path resolution: goes up one level from src/ to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading processed datasets...")
train_df = pd.read_csv(os.path.join(CLEANED_DIR, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(CLEANED_DIR, "test_processed.csv"))

# Separate features (X), target (y), and encounter IDs
target_col = "readmit_30d"
id_col = "encounter_id"

X = train_df.drop(columns=[id_col, target_col], errors="ignore")
y = train_df[target_col]
X_test = test_df.drop(columns=[id_col], errors="ignore")

# Ensure test columns match train columns exactly
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# CRITICAL FIX: Keep only numeric and boolean columns to prevent string/ID dtype errors in LightGBM
X = X.select_dtypes(include=[np.number, "bool"])
X_test = X_test.select_dtypes(include=[np.number, "bool"])

# Handle remaining NaNs (Fill numerical NaNs with -999)
X = X.fillna(-999)
X_test = X_test.fillna(-999)

print(f"Training on {X.shape[1]} clean numeric features.")

# Setup Stratified K-Fold Cross-Validation (preserves the 10% class ratio per fold)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Recommended Model: LightGBM Classifier for high-speed, high-performance tabular learning
model = LGBMClassifier(
    n_estimators=781,
    learning_rate=0.011775347352758416,
    max_depth=9,
    num_leaves=17,
    min_child_samples = 71,
    colsample_bytree = 0.8033653656362315,
    reg_alpha = 0.0015119140282697282,
    reg_lambda = 1.0199841615882324
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Stratified Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict out-of-fold validation probabilities
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Accumulate test predictions across folds
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC-AUC: {overall_auc:.4f}")

# Generate Submission File in the root models directory
submission = pd.DataFrame({
    "encounter_id": test_df["encounter_id"],
    "readmit_30d": test_preds
})

sub_path = os.path.join(MODELS_DIR, "submission_lgb.csv")
submission.to_csv(sub_path, index=False)
print(f"Submission file successfully saved to root folder: models/submission_lgb.csv")