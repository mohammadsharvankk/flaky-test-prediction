"""
Step 1 of the SHAP/Dashboard phase: produce the artifacts a dashboard needs.

1. Get honest out-of-fold (OOF) risk scores for every test, using the tuned
   Random Forest and the same LeaveOneGroupOut setup as before (each test's
   score comes from a model that never saw its own project during training).
2. Pick ONE global decision threshold that maximizes F1 across all OOF
   predictions pooled together - this is the single threshold the dashboard
   will use to color tests red/green, not a different one per project.
3. Fit a final "production" model on the FULL dataset (all 23 projects) -
   this is the model that would actually be deployed, since in production
   you're not holding any project out, you're using everything you have.
4. Save per-test risk scores (from OOF, for honesty about model quality)
   alongside the final model's predictions (for actual deployment) so the
   dashboard can show both.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered_v3.csv"
CHECKPOINT_PATH = "results/oof_scores_v3.csv"

with open("results/rf_best_params.json") as f:
    BEST_PARAMS = json.load(f)


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return df, X, y, groups


def compute_oof_scores(X, y, groups):
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

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **BEST_PARAMS)
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        for idx, prob, label in zip(test_idx, probs, y_test):
            new_rows.append({"row_index": idx, "project": held_out, "flaky": label, "oof_risk_score": prob})

        combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
        combined.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  [{fold_i}/{n_splits}] {held_out} done - checkpointed")

    return pd.read_csv(CHECKPOINT_PATH)


def pick_global_threshold(oof_df):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    y_true = oof_df["flaky"]
    probs = oof_df["oof_risk_score"]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (probs >= best_t).astype(int)
    print(f"\nGlobal threshold chosen: {best_t:.2f}")
    print(f"  Precision: {precision_score(y_true, preds, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, preds, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, preds, zero_division=0):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, probs):.4f}")
    return best_t


if __name__ == "__main__":
    df, X, y, groups = load_data()
    oof_df = compute_oof_scores(X, y, groups)
    if oof_df["project"].nunique() >= groups.nunique():
        threshold = pick_global_threshold(oof_df)
        with open("results/global_threshold_v3.json", "w") as f:
            json.dump({"threshold": threshold}, f)
        print("\nSaved global_threshold_v3.json")
    else:
        print(f"\n[NOT COMPLETE] {oof_df['project'].nunique()}/{groups.nunique()} projects. Re-run to continue.")
