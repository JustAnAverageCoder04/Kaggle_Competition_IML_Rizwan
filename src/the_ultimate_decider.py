import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("Loading individual model submissions...")
lgb_sub = pd.read_csv(os.path.join(MODELS_DIR, "submission_lgb.csv"))
xgb_sub = pd.read_csv(os.path.join(MODELS_DIR, "submission_xgb.csv"))

# Average the predicted probabilities (50% LightGBM + 50% XGBoost)
ensemble_sub = lgb_sub.copy()
ensemble_sub["readmit_30d"] = 0.5 * lgb_sub["readmit_30d"] + 0.5 * xgb_sub["readmit_30d"]

ensemble_path = os.path.join(MODELS_DIR, "submission_ensemble.csv")
ensemble_sub.to_csv(ensemble_path, index=False)
print(f"Ensemble submission successfully saved to: {ensemble_path}")