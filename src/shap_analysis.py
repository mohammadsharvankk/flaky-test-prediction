"""
Step 2: Fit the final production model (trained on ALL 23 projects, since a
deployed model isn't holding anything out) and run SHAP to explain it.

Uses TreeExplainer, which is exact and fast for tree-based models like
Random Forest (no approximation needed, unlike KernelExplainer).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"

with open("results/rf_best_params.json") as f:
    BEST_PARAMS = json.load(f)

df = pd.read_csv(DATA_PATH)
groups = df["project"]
y = df["flaky"]
X = df.drop(columns=["project", "flaky"])
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("Fitting final production model on all", len(df), "rows...")
final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **BEST_PARAMS)
final_model.fit(X_s, y)

# Save the fitted model + scaler for reuse (dashboard needs these to score new tests)
import pickle
with open("models/final_model.pkl", "wb") as f:
    pickle.dump({"model": final_model, "scaler": scaler, "feature_names": feature_names}, f)
print("Saved final_model.pkl")

print("Computing SHAP values (TreeExplainer) on a representative sample...")
# Full TreeExplainer over all 21,661 rows is too slow on a single core.
# Sample instead: keep ALL flaky examples (only 802, so keep every one) plus
# a random sample of non-flaky tests, for a representative but tractable set.
rng = np.random.RandomState(RANDOM_STATE)
flaky_idx = df.index[y == 1].to_numpy()
nonflaky_idx = df.index[y == 0].to_numpy()
sample_nonflaky = rng.choice(nonflaky_idx, size=min(800, len(nonflaky_idx)), replace=False)
sample_idx = np.concatenate([flaky_idx, sample_nonflaky])
sample_idx.sort()

X_sample_s = X_s[sample_idx]
print(f"Explaining {len(sample_idx)} rows ({len(flaky_idx)} flaky + {len(sample_nonflaky)} non-flaky sample)...")

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample_s)

# shap_values shape handling: for binary classification RandomForest,
# newer shap versions return array of shape (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    sv_flaky = shap_values[1]  # class 1 = flaky
elif shap_values.ndim == 3:
    sv_flaky = shap_values[:, :, 1]
else:
    sv_flaky = shap_values

# --- Global feature importance: mean absolute SHAP value per feature ---
mean_abs_shap = np.abs(sv_flaky).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv("results/shap_feature_importance.csv", index=False)

print("\nTop 15 features driving flaky predictions:")
print(importance_df.head(15).to_string(index=False))

# --- Per-test SHAP values, saved for the dashboard to use ---
shap_df = pd.DataFrame(sv_flaky, columns=[f"shap_{c}" for c in feature_names])
shap_df["row_index"] = sample_idx
shap_df.to_csv("results/shap_values_per_test.csv", index=False)
print("\nSaved shap_feature_importance.csv and shap_values_per_test.csv")
