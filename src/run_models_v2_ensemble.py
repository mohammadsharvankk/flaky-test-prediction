"""
Flaky Test Prediction — Improved Ensemble (targeting ROC-AUC > 0.708)

Prior run showed: RandomForest (0.708) and GradientBoosting (0.698) are the
strongest individual models. LinearSVM and LogisticRegression (~0.63) were
dragging down the naive 5-model Stacking/Voting ensembles. This version:

1. Selective ensemble: only combine the top 3 models by ROC-AUC
   (RandomForest, GradientBoosting, XGBoost) - drop the weak linear models.
2. Stacking with passthrough=True: the meta-learner sees the original
   features AND the base model predictions, not just the predictions alone.
   This often recovers signal that gets lost when compressing to just
   base-model outputs.
3. Weighted soft voting: instead of a plain average, weight each model's
   probability by its own individual ROC-AUC, so the stronger models get
   proportionally more say.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"
CHECKPOINT_PATH = "results/fold_results_v2_ensemble.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


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

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        top_models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            "GradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                n_jobs=-1, scale_pos_weight=pos_ratio, random_state=RANDOM_STATE,
            ),
        }

        fold_rows = []
        fitted_probs = {}
        for name, model in top_models.items():
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
            fitted_probs[name] = probs
            preds = (probs >= 0.5).astype(int)
            fold_rows.append({
                "project": held_out, "model": name, "n_test": len(y_test),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probs),
            })

        # --- Weighted soft voting: weight by each model's OOF training ROC-AUC ---
        # (approximated here via a quick internal holdout on training data)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx2, val_idx2 = next(gss.split(X_train_s, y_train, groups.iloc[train_idx]))
        weights = {}
        for name, model in top_models.items():
            m2 = type(model)(**model.get_params())
            m2.fit(X_train_s[tr_idx2], y_train.iloc[tr_idx2])
            val_probs = m2.predict_proba(X_train_s[val_idx2])[:, 1]
            try:
                weights[name] = max(roc_auc_score(y_train.iloc[val_idx2], val_probs) - 0.5, 0.01)
            except ValueError:
                weights[name] = 0.33
        total_w = sum(weights.values())
        weighted_probs = sum(fitted_probs[n] * (weights[n] / total_w) for n in top_models)
        preds = (weighted_probs >= 0.5).astype(int)
        fold_rows.append({
            "project": held_out, "model": "WeightedVoting", "n_test": len(y_test),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, weighted_probs),
        })

        # --- Stacking with passthrough (meta-learner sees raw features + base predictions) ---
        stack = StackingClassifier(
            estimators=list(top_models.items()),
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=3, n_jobs=-1, passthrough=True,
        )
        stack.fit(X_train_s, y_train)
        probs = stack.predict_proba(X_test_s)[:, 1]
        preds = stack.predict(X_test_s)
        fold_rows.append({
            "project": held_out, "model": "StackedPassthrough", "n_test": len(y_test),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })

        new_rows.extend(fold_rows)
        combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
        combined.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  [{fold_i}/{n_splits}] {held_out} done - checkpointed")

    return pd.read_csv(CHECKPOINT_PATH)


def summarize(df):
    print("\n" + "=" * 70)
    summary = df.groupby("model")[["precision", "recall", "f1", "roc_auc"]].mean().sort_values("roc_auc", ascending=False)
    print(summary.to_string())
    summary.to_csv("results/model_summary_v2_ensemble.csv")


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate(X, y, groups)
    if df["project"].nunique() >= groups.nunique():
        summarize(df)
    else:
        print(f"\n[NOT COMPLETE] {df['project'].nunique()}/{groups.nunique()} projects. Re-run to continue.")
