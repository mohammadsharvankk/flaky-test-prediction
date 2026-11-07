"""
Tests whether the 9 new project-relative features (v3) improve ROC-AUC over
the original 37-feature set (v2), using the identical tuned Random Forest
hyperparameters and Leave-One-Group-Out methodology for a fair comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
CHECKPOINT_PATH = "results/feature_v3_test_results.csv"

with open("results/rf_best_params.json") as f:
    RF_PARAMS = json.load(f)


def load_data(path):
    df = pd.read_csv(path)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


def evaluate(X, y, groups, feature_set_name):
    logo = LeaveOneGroupOut()
    try:
        done_df = pd.read_csv(CHECKPOINT_PATH)
        completed = set(zip(done_df["feature_set"], done_df["project"]))
    except FileNotFoundError:
        done_df = None
        completed = set()

    n_splits = logo.get_n_splits(X, y, groups)
    new_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        held_out = groups.iloc[test_idx].iloc[0]
        if (feature_set_name, held_out) in completed:
            continue

        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **RF_PARAMS)
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)

        new_rows.append({
            "feature_set": feature_set_name, "project": held_out, "n_test": len(y_test),
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

        combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
        combined.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  [{feature_set_name}] [{fold_i}/{n_splits}] {held_out} done")

    return pd.read_csv(CHECKPOINT_PATH)


def summarize(df):
    print("\n" + "=" * 70)
    summary = df.groupby("feature_set")[["roc_auc", "pr_auc", "precision", "recall", "f1"]].mean()
    print(summary.round(4).to_string())
    summary.to_csv("results/feature_v3_test_summary.csv")


if __name__ == "__main__":
    X_v2, y_v2, groups_v2 = load_data("data/flakeflagger_engineered.csv")
    df = evaluate(X_v2, y_v2, groups_v2, "v2_original_37_features")

    X_v3, y_v3, groups_v3 = load_data("data/flakeflagger_engineered_v3.csv")
    df = evaluate(X_v3, y_v3, groups_v3, "v3_with_project_relative_features")

    n_expected = groups_v2.nunique() * 2
    if len(df) >= n_expected * 0.9:
        summarize(df)
    else:
        print(f"\n[NOT COMPLETE] {len(df)}/{n_expected}. Re-run to continue.")
