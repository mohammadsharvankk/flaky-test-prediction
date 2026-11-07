"""
Random split vs. cross-project split: same model, same features, different answer.

This script is the core evidence for the project's main finding. It trains the
tuned Random Forest on the v3 (46-feature) FlakeFlagger set twice:

  1. Random split  - stratified 5-fold CV with shuffling. Tests from the same
                     project land in both train and test folds, which is how
                     much of the published work evaluates.
  2. Cross-project - Leave-One-Group-Out by project. The model never sees the
                     held-out project during training, which is what happens
                     when you point a flaky-test predictor at a new codebase.

The gap between the two ROC-AUC numbers is the part of the "accuracy" that
comes from recognising projects rather than learning what makes a test flaky.

Usage (from the repo root):
    python src/split_comparison.py

Output: results/split_comparison.csv
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered_v3.csv"
OUT_PATH = "results/split_comparison.csv"

with open("results/rf_best_params.json") as f:
    BEST_PARAMS = json.load(f)


def fit_score(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    return roc_auc_score(y_test, probs), average_precision_score(y_test, probs)


def evaluate(name, splits, X, y, groups=None):
    rows = []
    for i, (tr, te) in enumerate(splits, 1):
        if y.iloc[te].nunique() < 2:
            # A project with no flaky tests has no defined ROC-AUC; skip it,
            # exactly as the main LOGO experiments do.
            continue
        auc, ap = fit_score(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
        fold = groups.iloc[te].iloc[0] if groups is not None else f"fold_{i}"
        rows.append({"split": name, "fold": fold, "roc_auc": auc, "pr_auc": ap})
        print(f"  {name:<14} {fold:<28} ROC-AUC {auc:.3f}")
    return rows


def main():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])

    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows += evaluate("random_5fold", skf.split(X, y), X, y)
    rows += evaluate("cross_project", LeaveOneGroupOut().split(X, y, groups), X, y, groups)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_PATH, index=False)

    summary = res.groupby("split")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).round(3)
    print("\n", summary)
    gap = res[res.split == "random_5fold"].roc_auc.mean() - res[res.split == "cross_project"].roc_auc.mean()
    print(f"\nROC-AUC gap (random - cross-project): {gap:.3f}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
