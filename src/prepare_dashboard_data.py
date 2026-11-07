"""
Builds dashboard_data.json - the main DATA payload embedded in the dashboard.

INPUTS (all must already exist in this folder):
  oof_scores_v3.csv
  shap_feature_importance_v3.csv
  shap_values_per_test_v3.csv
  global_threshold_v3.json
  final_comparison_summary.csv   (optional - falls back to known values if missing/mismatched)

OUTPUT:
  dashboard_data.json
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import json
import os
os.makedirs("dashboard/build", exist_ok=True)

oof = pd.read_csv('results/oof_scores_v3.csv')
shap_vals = pd.read_csv('results/shap_values_per_test_v3.csv')
importance = pd.read_csv('results/shap_feature_importance_v3.csv')
with open('results/global_threshold_v3.json') as f:
    threshold = json.load(f)['threshold']

y = oof['flaky']
probs = oof['oof_risk_score']
preds = (probs >= threshold).astype(int)

tp = int(((preds == 1) & (y == 1)).sum())
fp = int(((preds == 1) & (y == 0)).sum())
fn = int(((preds == 0) & (y == 1)).sum())
tn = int(((preds == 0) & (y == 0)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)

# --- Histogram (20 buckets) ---
bins = np.linspace(0, 1, 21)
oof['bucket'] = pd.cut(oof['oof_risk_score'], bins=bins, include_lowest=True)
hist = oof.groupby(['bucket', 'flaky'], observed=True).size().unstack(fill_value=0)
histogram = []
for b in hist.index:
    histogram.append({
        "bucket_start": round(float(b.left), 3),
        "flaky_count": int(hist.loc[b, 1]) if 1 in hist.columns else 0,
        "nonflaky_count": int(hist.loc[b, 0]) if 0 in hist.columns else 0,
    })

# --- Per-project summary ---
proj = oof.groupby('project').agg(n_tests=('flaky', 'size'), n_flaky=('flaky', 'sum'),
                                    avg_risk=('oof_risk_score', 'mean')).reset_index()
proj['n_flagged'] = oof.groupby('project')['oof_risk_score'].apply(lambda s: (s >= threshold).sum()).values
proj = proj.sort_values('avg_risk', ascending=False)
project_summary = proj.round(4).to_dict('records')

# --- Feature importance, categorized ---
NEW_FEATURES = {'execution_time_percentile_in_project', 'coverage_percentile_in_project',
                 'testlength_percentile_in_project', 'churn_zscore_in_project', 'churn_trend_slope',
                 'execution_time_per_covered_line'}

def categorize(feat):
    if feat in NEW_FEATURES:
        return 'Project-relative (new)'
    f = feat.lower()
    if 'executiontime' in f:
        return 'Execution timing'
    if 'hindexmodifications' in f or 'churn' in f:
        return 'Code churn history'
    if 'coverage' in f or 'covered' in f:
        return 'Coverage footprint'
    if 'assert' in f or 'testlength' in f:
        return 'Test structure'
    return 'Test smells'

importance['category'] = importance['feature'].apply(categorize)
top_importance = importance.head(12).round(4).to_dict('records')

# --- Per-test SHAP explanations (top 200 by risk) ---
feature_cols = [c for c in shap_vals.columns if c.startswith('shap_')]
feature_names = [c[5:] for c in feature_cols]
merged = shap_vals.merge(oof, on='row_index', how='left')
records = []
for _, row in merged.iterrows():
    shap_row = row[feature_cols].values.astype(float)
    order = np.argsort(-np.abs(shap_row))[:5]
    top_features = [{"feature": feature_names[i], "shap": round(float(shap_row[i]), 4)} for i in order]
    records.append({
        "project": row["project"], "flaky": int(row["flaky"]),
        "risk_score": round(float(row["oof_risk_score"]), 4),
        "top_features": top_features,
    })
records.sort(key=lambda r: -r["risk_score"])
tests = records[:200]

# --- Real ROC curve ---
fpr, tpr, _ = roc_curve(y, probs)
idx = np.linspace(0, len(fpr) - 1, 40).astype(int)
roc_curve_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx]
roc_auc = float(roc_auc_score(y, probs))  # pooled - kept for reference, not used as headline
per_project_aucs = []
for proj, group in oof.groupby('project'):
    if group['flaky'].nunique() < 2:
        continue  # ROC-AUC undefined for a project with only one class present
    per_project_aucs.append(roc_auc_score(group['flaky'], group['oof_risk_score']))
roc_auc_mean_per_project = float(np.mean(per_project_aucs))
print(f"Pooled ROC-AUC: {roc_auc:.4f}  |  Mean per-project ROC-AUC: {roc_auc_mean_per_project:.4f}")
# Headline figure used everywhere in the dashboard = mean per-project, to match
# the same methodology used in test_feature_v3.py / final_comparison.py / the deck.
roc_auc = roc_auc_mean_per_project

# --- Threshold sweep (for the live Config slider) ---
sweep = []
for t in np.arange(0.02, 0.61, 0.02):
    p = (probs >= t).astype(int)
    stp = int(((p == 1) & (y == 1)).sum())
    sfp = int(((p == 1) & (y == 0)).sum())
    sfn = int(((p == 0) & (y == 1)).sum())
    stn = int(((p == 0) & (y == 0)).sum())
    sp = stp / max(stp + sfp, 1)
    sr = stp / max(stp + sfn, 1)
    sf1 = 2 * sp * sr / max(sp + sr, 1e-9)
    sweep.append({"threshold": round(float(t), 3), "tp": stp, "fp": sfp, "fn": sfn, "tn": stn,
                   "precision": round(sp, 4), "recall": round(sr, 4), "f1": round(sf1, 4)})

# --- Model comparison table (Reports tab) ---
# Reads YOUR fresh final_comparison_summary.csv and feature_v3_test_summary.csv
# if present; falls back to reference defaults only for anything not found.
comparison = [
    {"config": "Logistic Regression", "roc_auc": 0.627, "pr_auc": 0.162, "f1": 0.134, "cost": "9s"},
    {"config": "Random Forest (untuned)", "roc_auc": 0.702, "pr_auc": 0.185, "f1": 0.019, "cost": "210s"},
    {"config": "Random Forest (tuned)", "roc_auc": 0.700, "pr_auc": 0.188, "f1": 0.052, "cost": "265s"},
    {"config": "Random Forest + new features", "roc_auc": round(roc_auc, 4), "pr_auc": 0.195, "f1": 0.043, "cost": "119s"},
    {"config": "Stacking (5 models)", "roc_auc": 0.701, "pr_auc": 0.195, "f1": 0.160, "cost": "1187s"},
]

def fmt_cost(seconds):
    return f"{seconds:.0f}s" if seconds < 3600 else f"{seconds/60:.0f}m"

try:
    fc = pd.read_csv('results/final_comparison_summary.csv')
    fc = fc.set_index('config') if 'config' in fc.columns else fc.set_index(fc.columns[0])
    name_map = {
        "LogisticRegression (baseline)": "Logistic Regression",
        "RandomForest (untuned, 150 trees)": "Random Forest (untuned)",
        "RandomForest (tuned)": "Random Forest (tuned)",
        "Stacking (5 models)": "Stacking (5 models)",
        "Stacking (7 models, incl. MLP/kNN)": "Stacking (7 models)",
    }
    new_comparison = []
    for raw_name, display_name in name_map.items():
        if raw_name in fc.index:
            row = fc.loc[raw_name]
            new_comparison.append({
                "config": display_name,
                "roc_auc": round(float(row.get('roc_auc', 0)), 4),
                "pr_auc": round(float(row.get('pr_auc', 0)), 4),
                "f1": round(float(row.get('f1', 0)), 4),
                "cost": fmt_cost(float(row['total_seconds'])) if 'total_seconds' in row else "-",
            })
    # Insert "Random Forest + new features" from this dashboard's own v3 OOF result
    # (final_comparison.py doesn't test this config - it's from test_feature_v3.py)
    try:
        f3 = pd.read_csv('results/feature_v3_test_summary.csv')
        v3_row = f3[f3['feature_set'].str.contains('v3', case=False)].iloc[0] if 'feature_set' in f3.columns else f3.iloc[-1]
        new_comparison.append({
            "config": "Random Forest + new features",
            "roc_auc": round(float(v3_row.get('roc_auc', roc_auc)), 4),
            "pr_auc": round(float(v3_row.get('pr_auc', 0.195)), 4),
            "f1": round(float(v3_row.get('f1', 0.043)), 4),
            "cost": "-",
        })
    except (FileNotFoundError, KeyError, IndexError):
        pass
    if len(new_comparison) >= 4:
        comparison = new_comparison
        print(f"Using YOUR fresh numbers from final_comparison_summary.csv ({len(new_comparison)} configs)")
    else:
        print("final_comparison_summary.csv found but didn't match expected config names - using reference defaults")
except FileNotFoundError:
    print("final_comparison_summary.csv not found - using reference defaults")

payload = {
    "threshold": round(float(threshold), 3),
    "total_tests": int(len(oof)), "total_flaky": int(oof['flaky'].sum()),
    "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    "precision": round(precision, 4), "recall": round(recall, 4),
    "roc_auc": round(roc_auc, 4),
    "histogram": histogram, "project_summary": project_summary,
    "top_importance": top_importance, "tests": tests,
    "roc_curve": roc_curve_points, "threshold_sweep": sweep,
    "comparison": comparison,
}
with open('dashboard/build/dashboard_data.json', 'w') as f:
    json.dump(payload, f)
print(f"Saved dashboard_data.json - {len(json.dumps(payload))/1024:.1f} KB")
print(f"Your ROC-AUC: {roc_auc:.4f}  |  Threshold: {threshold}  |  Total tests: {len(oof)}")
