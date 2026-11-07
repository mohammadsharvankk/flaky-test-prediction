"""
Diverse Ensemble Test — can adding genuinely different model types beat
Random Forest alone (0.708-0.721 ROC-AUC)?

Previous ensembles (Voting/Stacking of RF+GB+XGBoost) failed to beat RF
alone because all three are tree-based and make similar mistakes. This
test adds two fundamentally different model types:
  - MLP (neural network): learns different decision boundaries than trees
  - k-NN: purely distance-based, no tree structure at all

If these help, it confirms the diversity hypothesis. If not, it tells us
Random Forest alone is close to the ceiling for this feature set.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"
CHECKPOINT_PATH = "results/diverse_ensemble_results.csv"

with open("results/rf_best_params.json") as f:
    RF_PARAMS = json.load(f)


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

        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        models = {
            "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **RF_PARAMS),
            "GradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                n_jobs=1, scale_pos_weight=pos_ratio, random_state=RANDOM_STATE,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1),
        }

        fold_rows = []
        fitted_probs = {}
        for name, model in models.items():
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

        # Plain soft voting across all 5 diverse models
        vote_probs = np.mean(list(fitted_probs.values()), axis=0)
        preds = (vote_probs >= 0.5).astype(int)
        fold_rows.append({
            "project": held_out, "model": "DiverseVoting", "n_test": len(y_test),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, vote_probs),
        })

        # Stacking across all 5 diverse models, with passthrough
        stack = StackingClassifier(
            estimators=list(models.items()),
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=3, n_jobs=1, passthrough=True,
        )
        stack.fit(X_train_s, y_train)
        probs = stack.predict_proba(X_test_s)[:, 1]
        preds = stack.predict(X_test_s)
        fold_rows.append({
            "project": held_out, "model": "DiverseStacked", "n_test": len(y_test),
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
    summary.to_csv("results/diverse_ensemble_summary.csv")


if __name__ == "__main__":
    X, y, groups = load_data()
    df = evaluate(X, y, groups)
    if df["project"].nunique() >= groups.nunique():
        summarize(df)
    else:
        print(f"\n[NOT COMPLETE] {df['project'].nunique()}/{groups.nunique()} projects. Re-run to continue.")
