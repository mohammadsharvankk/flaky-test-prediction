"""
Exports the trained Random Forest (all 300 trees) + scaler + medians + two
real example rows, for the dashboard's in-browser "Predict" tab.

INPUTS:
  final_model_v3.pkl              (dict with keys: model, scaler, feature_names)
  flakeflagger_engineered_v3.csv

OUTPUTS:
  predictor_trees.json   (feature_names, scaler_mean, scaler_scale, trees)
  predictor_ui.json      (medians, smell_cols, raw_numeric, churn_windows_all,
                           flaky_example, clean_example)
"""
import pickle
import json
import os
os.makedirs("dashboard/build", exist_ok=True)
import pandas as pd

with open('models/final_model_v3.pkl', 'rb') as f:
    d = pickle.load(f)
model, scaler, feature_names = d['model'], d['scaler'], d['feature_names']

df = pd.read_csv('data/flakeflagger_engineered_v3.csv')

SMELL_COLS = ['assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',
              'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war']
RAW_NUMERIC = ['testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime',
               'projectSourceLinesCovered', 'projectSourceClassesCovered', 'num_third_party_libs',
               'hIndexModificationsPerCoveredLine_window100']
CHURN_WINDOWS_ALL = ['hIndexModificationsPerCoveredLine_window5', 'hIndexModificationsPerCoveredLine_window10',
                      'hIndexModificationsPerCoveredLine_window25', 'hIndexModificationsPerCoveredLine_window50',
                      'hIndexModificationsPerCoveredLine_window75', 'hIndexModificationsPerCoveredLine_window100',
                      'hIndexModificationsPerCoveredLine_window500', 'hIndexModificationsPerCoveredLine_window10000']

medians = df[feature_names].median().to_dict()

flaky_examples = df[df['flaky'] == 1]
clean_examples = df[df['flaky'] == 0]
flaky_row = flaky_examples.iloc[len(flaky_examples) // 3][SMELL_COLS + RAW_NUMERIC].to_dict()
clean_row = clean_examples.iloc[len(clean_examples) // 3][SMELL_COLS + RAW_NUMERIC].to_dict()


def export_tree(tree):
    t = tree.tree_
    probs = t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1)
    return {
        "feature": t.feature.tolist(), "threshold": [round(float(x), 5) for x in t.threshold],
        "left": t.children_left.tolist(), "right": t.children_right.tolist(),
        "prob": [round(float(x), 4) for x in probs],
    }


print(f"Exporting {model.n_estimators} trees ...")
trees = [export_tree(e) for e in model.estimators_]

predictor_trees = {
    "feature_names": feature_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "trees": trees,
}
with open('dashboard/build/predictor_trees.json', 'w') as f:
    json.dump(predictor_trees, f)
print(f"Saved predictor_trees.json - {len(json.dumps(predictor_trees)) / 1024 / 1024:.2f} MB")

predictor_ui = {
    "medians": {k: round(float(v), 4) for k, v in medians.items()},
    "smell_cols": SMELL_COLS,
    "raw_numeric": RAW_NUMERIC,
    "churn_windows_all": CHURN_WINDOWS_ALL,
    "flaky_example": {k: round(float(v), 4) for k, v in flaky_row.items()},
    "clean_example": {k: round(float(v), 4) for k, v in clean_row.items()},
}
with open('dashboard/build/predictor_ui.json', 'w') as f:
    json.dump(predictor_ui, f)
print(f"Saved predictor_ui.json - {len(json.dumps(predictor_ui)) / 1024:.1f} KB")
