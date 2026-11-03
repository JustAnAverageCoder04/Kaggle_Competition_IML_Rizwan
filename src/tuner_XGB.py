import os
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Suppress excessive Optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading processed datasets for Optuna tuning...")
train_df = pd.read_csv(os.path.join(CLEANED_DIR, "train_processed.csv"))

target_col = "readmit_30d"
id_col = "encounter_id"

X = train_df.drop(columns=[id_col, target_col], errors="ignore")
y = train_df[target_col]

# Ensure clean numeric/boolean types and handle NaNs
#X = X.select_dtypes(include=[np.number, "bool"]).fillna(-999)
X = X.select_dtypes(include=[np.number, "bool"])

def objective(trial):
    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), # Replaces min_child_samples
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": 9,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    }

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y, groups=train_df["hospital_id"]):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Return Out-of-Fold ROC-AUC to be maximized
    return roc_auc_score(y, oof_preds)

print("Starting Optuna study (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n--- Tuning Complete ---")
print(f"Best OOF ROC-AUC: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save best parameters to a dictionary or directly use them in your training script
best_params = study.best_params