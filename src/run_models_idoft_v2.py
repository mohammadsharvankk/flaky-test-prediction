"""
Flaky Test Prediction — IDoFT (code-based) Benchmarking
Dataset: Flakify_IDoFT_dataset (engineered) — 3,862 tests, 312 projects, ~84.6% flaky

NOTE ON DATASET CHOICE: this is a different dataset from the FlakeFlagger run.
It's far less imbalanced (84.6% flaky vs. FlakeFlagger's 3.7%), which produces
meaningfully better precision/recall/F1 numbers. That's a real, honest effect
of a more balanced problem - be transparent that this is a different dataset
with a different (curated, catalog-style) composition, not the same problem
solved better.

Evaluation strategy: StratifiedGroupKFold (5 folds) grouping by `project`,
NOT LeaveOneGroupOut. With 312 projects and 71 of them having only a single
test, one-fold-per-project would leave most folds with too little data to
evaluate meaningfully (some would be a single row). Grouping into 5 folds
still guarantees no project appears in both train and test in the same fold
(no leakage), while keeping each fold's test set large enough to be stable.

Models (5): Logistic Regression, Linear SVM, Random Forest,
Hist Gradient Boosting, XGBoost — same as the FlakeFlagger run, for
consistency across both benchmarks.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/idoft_engineered_v2.csv"
N_FOLDS = 5


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


def build_base_models(pos_ratio):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
            cv=3,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="logloss", n_jobs=-1,
            scale_pos_weight=pos_ratio, random_state=RANDOM_STATE,
        ),
    }


def evaluate(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        base_models = build_base_models(pos_ratio)

        for name, model in base_models.items():
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]
            rows.append({
                "fold": fold_i, "model": name, "n_test": len(y_test),
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probs),
            })

        voter = VotingClassifier(estimators=list(base_models.items()), voting="soft", n_jobs=-1)
        voter.fit(X_train_s, y_train)
        preds = voter.predict(X_test_s)
        probs = voter.predict_proba(X_test_s)[:, 1]
        rows.append({
            "fold": fold_i, "model": "Voting", "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        stack = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=3, n_jobs=-1,
        )
        stack.fit(X_train_s, y_train)
        preds = stack.predict(X_test_s)
        probs = stack.predict_proba(X_test_s)[:, 1]
        rows.append({
            "fold": fold_i, "model": "Stacked", "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        print(f"  Fold {fold_i}/{N_FOLDS} done (n_test={len(y_test)})")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 80)
    print(f"{'Model':<20}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("=" * 80)
    summary = df.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean()
    summary = summary.sort_values("f1", ascending=False)
    for name, row in summary.iterrows():
        print(f"{name:<20}{row['accuracy']:>12.3f}{row['precision']:>12.3f}{row['recall']:>12.3f}{row['f1']:>12.3f}{row['roc_auc']:>12.3f}")
    print("=" * 80)
    best = summary["f1"].idxmax()
    print(f"\nBest model by mean F1 across {N_FOLDS} folds: {best}")
    summary.to_csv("results/idoft_model_summary_v2.csv")
    df.to_csv("results/idoft_fold_results_v2.csv", index=False)


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate(X, y, groups)
    summarize(df)
