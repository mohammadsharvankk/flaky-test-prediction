"""
Threshold Tuning for the tuned Random Forest (ROC-AUC 0.721 model)

The default 0.5 cutoff is a bad fit for ~3.7% flaky data - it's calibrated
for a 50/50 problem, not this one. But picking a threshold BY LOOKING AT
THE TEST FOLD would be leakage - identical mistake to picking hyperparameters
on the test set.

Honest approach per outer fold (one of the 23 held-out projects):
  1. Train on the 22 other projects.
  2. Get out-of-fold probability predictions on those SAME 22 training
     projects only, via an inner GroupKFold (still grouped by project, so
     no leakage even at this inner stage).
  3. Find the threshold that maximizes F1 on those inner out-of-fold
     predictions - this never touches the actual held-out test project.
  4. Apply that chosen threshold to the real held-out project's predictions.

This means the threshold can differ slightly fold to fold (as it should -
it's re-derived honestly each time), and the final reported number reflects
what you'd actually get in practice, not an optimistic test-set-tuned one.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"
CHECKPOINT_PATH = "results/threshold_tuned_results.csv"

with open("results/rf_best_params.json") as f:
    BEST_PARAMS = json.load(f)


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


def best_f1_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(X, y, groups):
    logo = LeaveOneGroupOut()
    try:
        done_df = pd.read_csv(CHECKPOINT_PATH)
        completed = set(done_df["project"].unique())
        print(f"Resuming: {len(completed)} projects already scored.")
    except FileNotFoundError:
        done_df = None
        completed = set()

    n_splits = logo.get_n_splits(X, y, groups)
    new_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        held_out = groups.iloc[test_idx].iloc[0]
        if held_out in completed:
            continue

        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Inner CV (grouped) to get honest out-of-fold probabilities on
        # training data only - this is what the threshold gets picked from.
        n_inner_groups = groups_train.nunique()
        inner_splits = min(5, n_inner_groups)
        gkf = GroupKFold(n_splits=inner_splits)

        inner_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **BEST_PARAMS)
        oof_probs = cross_val_predict(
            inner_model, X_train_s, y_train, cv=gkf.split(X_train_s, y_train, groups_train),
            method="predict_proba", n_jobs=1,
        )[:, 1]

        chosen_threshold = best_f1_threshold(y_train, oof_probs)

        # Final model: fit on ALL 22 training projects, evaluate on the real held-out project
        final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **BEST_PARAMS)
        final_model.fit(X_train_s, y_train)
        test_probs = final_model.predict_proba(X_test_s)[:, 1]
        test_preds_default = (test_probs >= 0.5).astype(int)
        test_preds_tuned = (test_probs >= chosen_threshold).astype(int)

        new_rows.append({
            "project": held_out, "n_test": len(y_test), "chosen_threshold": chosen_threshold,
            "precision_default": precision_score(y_test, test_preds_default, zero_division=0),
            "recall_default": recall_score(y_test, test_preds_default, zero_division=0),
            "f1_default": f1_score(y_test, test_preds_default, zero_division=0),
            "precision_tuned": precision_score(y_test, test_preds_tuned, zero_division=0),
            "recall_tuned": recall_score(y_test, test_preds_tuned, zero_division=0),
            "f1_tuned": f1_score(y_test, test_preds_tuned, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_probs),
        })

        combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
        combined.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  [{fold_i}/{n_splits}] {held_out} - threshold={chosen_threshold:.2f} - checkpointed")

    return pd.read_csv(CHECKPOINT_PATH)


def summarize(df):
    print("\n" + "=" * 70)
    print("Default threshold (0.5):")
    print(df[["precision_default", "recall_default", "f1_default"]].mean())
    print("\nTuned per-fold threshold (mean chosen threshold: {:.2f}):".format(df["chosen_threshold"].mean()))
    print(df[["precision_tuned", "recall_tuned", "f1_tuned"]].mean())
    print("\nROC-AUC (threshold-independent, unchanged):", df["roc_auc"].mean().round(4))


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate(X, y, groups)
    if df["project"].nunique() >= groups.nunique():
        summarize(df)
    else:
        print(f"\n[NOT COMPLETE] {df['project'].nunique()}/{groups.nunique()} projects. Re-run to continue.")
