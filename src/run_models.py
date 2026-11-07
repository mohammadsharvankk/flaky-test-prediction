"""
Flaky Test Prediction — Baseline Model Benchmarking
Dataset: FlakeFlagger (engineered) — 21,661 tests, 23 projects, ~3.7% flaky

Evaluation strategy: LeaveOneGroupOut by `project`. This is the honest way
to test generalization here — flakiness patterns are project-specific, so a
random row-level split would leak project identity into train and test and
inflate every score. Grouping by project means each fold trains on 22
projects and tests on the 1 held-out project it has never seen.

Models (5, matching your supervisor's ask):
  1. Logistic Regression   - linear baseline, cheap, interpretable coefficients
  2. Linear SVM            - margin-based baseline, comparable to LR
  3. Random Forest         - non-linear, handles feature interactions
  4. Hist Gradient Boosting - sklearn's fast histogram-based boosting baseline
  5. XGBoost               - stronger boosting baseline, usually top performer

Note: LSTM is intentionally NOT included. This dataset has no per-test time
series (each row is one static feature snapshot per test), so an LSTM would
have nothing sequential to model over — it would just be a slower, harder to
explain MLP. If your supervisor specifically wants a temporal/sequential
model, that requires a different dataset (e.g. per-run pass/fail history
from IDoFT), which is a separate piece of work worth flagging to him rather
than quietly working around.

Also included: a Stacking ensemble (per your supervisor's request to
explore combining models), with a Logistic Regression meta-learner so the
whole pipeline stays interpretable for the SHAP layer later.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


CHECKPOINT_PATH = "results/fold_results.csv"


def evaluate_cross_project(X, y, groups):
    logo = LeaveOneGroupOut()

    # Resume support: skip projects already scored in a prior (possibly
    # interrupted) run, and append new fold results to the same checkpoint file.
    try:
        done_df = pd.read_csv(CHECKPOINT_PATH)
        completed_projects = set(done_df["project"].unique())
        print(f"Resuming: {len(completed_projects)} projects already scored.")
    except FileNotFoundError:
        done_df = None
        completed_projects = set()

    n_splits = logo.get_n_splits(X, y, groups)
    print(f"LeaveOneGroupOut across {n_splits} projects total.\n")

    new_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        held_out_project = groups.iloc[test_idx].iloc[0]

        if held_out_project in completed_projects:
            continue

        # Skip folds where the held-out project has only one class present
        # (ROC-AUC is undefined with a single class in y_test)
        if y_test.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        base_models = {
            "LogisticRegression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "LinearSVM": CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
                cv=2,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            "GradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "XGBoost": XGBClassifier(
                n_estimators=150, eval_metric="logloss", n_jobs=-1,
                scale_pos_weight=pos_ratio, random_state=RANDOM_STATE,
            ),
        }

        fold_rows = []
        for name, model in base_models.items():
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]
            fold_rows.append({
                "project": held_out_project, "model": name, "n_test": len(y_test),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probs),
            })

        # --- Stacking ensemble: combine base models via a Logistic Regression meta-learner ---
        stack = StackingClassifier(
            estimators=[(n, m) for n, m in base_models.items()],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=2,
            n_jobs=-1,
        )
        stack.fit(X_train_s, y_train)
        preds = stack.predict(X_test_s)
        probs = stack.predict_proba(X_test_s)[:, 1]
        fold_rows.append({
            "project": held_out_project, "model": "Stacked", "n_test": len(y_test),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        # --- Voting ensemble (soft voting = averaged predicted probabilities) ---
        # This is the literal "voting method" your supervisor asked about, distinct
        # from Stacking: Voting combines predictions with no learned meta-model, so
        # it's simpler and more robust to a single base model overfitting, at the
        # cost of not learning which base models to trust more (which Stacking does).
        voter = VotingClassifier(
            estimators=[(n, m) for n, m in base_models.items()],
            voting="soft",
            n_jobs=-1,
        )
        voter.fit(X_train_s, y_train)
        preds = voter.predict(X_test_s)
        probs = voter.predict_proba(X_test_s)[:, 1]
        fold_rows.append({
            "project": held_out_project, "model": "Voting", "n_test": len(y_test),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        new_rows.extend(fold_rows)

        # Checkpoint after every fold so a killed/timed-out run can resume
        combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
        combined.to_csv(CHECKPOINT_PATH, index=False)

        print(f"  [{fold_i}/{n_splits}] held out: {held_out_project} (n_test={len(y_test)}) done - checkpointed")

    final_df = pd.read_csv(CHECKPOINT_PATH)
    return final_df


def summarize(df):
    print("\n" + "=" * 70)
    print(f"{'Model':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("=" * 70)
    summary = df.groupby("model")[["precision", "recall", "f1", "roc_auc"]].mean()
    for name, row in summary.iterrows():
        print(f"{name:<20}{row['precision']:>12.3f}{row['recall']:>12.3f}{row['f1']:>12.3f}{row['roc_auc']:>12.3f}")
    print("=" * 70)
    best = summary["f1"].idxmax()
    print(f"\nBest model by mean F1 across {df['project'].nunique()} held-out projects: {best}")
    summary.to_csv("results/model_summary.csv")


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate_cross_project(X, y, groups)
    n_expected = groups.nunique()
    n_done = df["project"].nunique()
    if n_done < n_expected:
        print(f"\n[NOT COMPLETE] {n_done}/{n_expected} projects scored so far. Re-run this script to continue.")
    else:
        summarize(df)
