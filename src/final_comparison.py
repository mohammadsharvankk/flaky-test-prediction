"""
Final Comparison Table — per your supervisor's request.

Adds PR-AUC (precision-recall AUC) alongside ROC-AUC, since PR-AUC is more
informative than ROC-AUC when the positive class is rare (~3.7% flaky here).
ROC-AUC can look deceptively good under heavy imbalance because it's diluted
by the huge number of true negatives; PR-AUC focuses specifically on how
well the model handles the positive (flaky) class, which is what actually
matters for this problem.

Also reports computational cost (total fit+predict wall-clock time across
all 23 folds), so the comparison table can show the accuracy/cost tradeoff,
not just accuracy alone.

A concise, representative set of configurations - not every model ever
tried - covering: a simple baseline, the untuned starting point, the best
single tuned model, and the two ensemble strategies actually tested.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"
CHECKPOINT_PATH = "results/final_comparison_results.csv"

with open("results/rf_best_params.json") as f:
    RF_TUNED_PARAMS = json.load(f)


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


def build_configs(pos_ratio):
    """Returns dict of config_name -> either a single estimator, or a
    callable that builds the full set of base models for an ensemble."""
    base_for_ensemble = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE), cv=2),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **RF_TUNED_PARAMS),
        "GradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss", n_jobs=1, scale_pos_weight=pos_ratio, random_state=RANDOM_STATE),
    }
    diverse_models = dict(base_for_ensemble)
    diverse_models["MLP"] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=RANDOM_STATE)
    diverse_models["kNN"] = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1)

    return {
        "LogisticRegression (baseline)": ("single", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "RandomForest (untuned, 150 trees)": ("single", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        "RandomForest (tuned)": ("single", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **RF_TUNED_PARAMS)),
        "Stacking (5 models)": ("stack", base_for_ensemble),
        "Stacking (7 models, incl. MLP/kNN)": ("stack", diverse_models),
    }


def evaluate(X, y, groups):
    logo = LeaveOneGroupOut()
    try:
        done_df = pd.read_csv(CHECKPOINT_PATH)
        completed = set(zip(done_df["config"], done_df["project"]))
        print(f"Resuming: {len(completed)} (config, project) pairs already scored.")
    except FileNotFoundError:
        done_df = None
        completed = set()

    n_splits = logo.get_n_splits(X, y, groups)
    new_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        held_out = groups.iloc[test_idx].iloc[0]
        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        configs = build_configs(pos_ratio)
        fold_rows = []

        for config_name, (kind, spec) in configs.items():
            if (config_name, held_out) in completed:
                continue

            t0 = time.time()
            if kind == "single":
                model = spec
                model.fit(X_train_s, y_train)
                probs = model.predict_proba(X_test_s)[:, 1]
            else:  # stack
                stack = StackingClassifier(
                    estimators=list(spec.items()),
                    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                    cv=3, n_jobs=1,
                )
                stack.fit(X_train_s, y_train)
                probs = stack.predict_proba(X_test_s)[:, 1]
            elapsed = time.time() - t0

            preds = (probs >= 0.5).astype(int)
            fold_rows.append({
                "config": config_name, "project": held_out, "n_test": len(y_test),
                "roc_auc": roc_auc_score(y_test, probs),
                "pr_auc": average_precision_score(y_test, probs),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "fit_predict_seconds": round(elapsed, 3),
            })

        if fold_rows:
            new_rows.extend(fold_rows)
            combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
            combined.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  [{fold_i}/{n_splits}] {held_out} done")

    return pd.read_csv(CHECKPOINT_PATH)


def summarize(df):
    print("\n" + "=" * 100)
    summary = df.groupby("config").agg(
        roc_auc=("roc_auc", "mean"),
        pr_auc=("pr_auc", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        total_seconds=("fit_predict_seconds", "sum"),
    ).sort_values("roc_auc", ascending=False)
    print(summary.round(4).to_string())
    summary.to_csv("results/final_comparison_summary.csv")


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate(X, y, groups)
    n_expected = build_configs(1.0).keys()
    n_pairs_expected = len(n_expected) * groups.nunique()
    if len(df) >= n_pairs_expected * 0.9:  # allow for skipped single-class folds
        summarize(df)
    else:
        print(f"\n[NOT COMPLETE] {len(df)}/{n_pairs_expected} (config,project) pairs. Re-run to continue.")
