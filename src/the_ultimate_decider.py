import os
import pandas as pd
from scipy.stats import rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("Loading individual model submissions...")
lgb_sub = pd.read_csv(os.path.join(MODELS_DIR, "submission_lgb.csv"))
xgb_sub = pd.read_csv(os.path.join(MODELS_DIR, "submission_xgb.csv"))

# Convert raw test probabilities to percentile ranks (0 to 1 scale)
lgb_ranks = rankdata(lgb_sub["readmit_30d"]) / len(lgb_sub)
xgb_ranks = rankdata(xgb_sub["readmit_30d"]) / len(xgb_sub)

# Blend the ranks instead of raw probabilities (50% LightGBM + 50% XGBoost)
ensemble_ranks = 0.5 * lgb_ranks + 0.5 * xgb_ranks

# Build the final ensemble submission dataframe
ensemble_sub = lgb_sub.copy()
ensemble_sub["readmit_30d"] = ensemble_ranks

ensemble_path = os.path.join(MODELS_DIR, "submission_ensemble.csv")
ensemble_sub.to_csv(ensemble_path, index=False)
print(f"Rank-averaged ensemble submission successfully saved to: {ensemble_path}")
print(f"Ensemble submission successfully saved to: {ensemble_path}")