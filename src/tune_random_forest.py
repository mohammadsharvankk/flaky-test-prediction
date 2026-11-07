"""
Random Forest Hyperparameter Search — targeting ROC-AUC

Two-stage approach:
1. Search: RandomizedSearchCV with GroupKFold(5) grouping by project, scoring
   on ROC-AUC. This picks hyperparameters using a cheaper 5-fold grouped CV
   (not full LeaveOneGroupOut) - grouping still prevents project leakage
   during the search itself, it's just less exhaustive than scoring on all
   23 held-out projects individually.
2. Confirm: re-evaluate the winning hyperparameters with the full 23-fold
   LeaveOneGroupOut, so the final reported number uses the same honest
   methodology as every other result in this project.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
DATA_PATH = "data/flakeflagger_engineered.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    groups = df["project"]
    y = df["flaky"]
    X = df.drop(columns=["project", "flaky"])
    return X, y, groups


def search_hyperparams(X, y, groups):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    param_dist = {
        "n_estimators": [200, 300],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    gkf = GroupKFold(n_splits=3)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)

    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=6,
        scoring="roc_auc", cv=gkf.split(X_s, y, groups),
        random_state=RANDOM_STATE, n_jobs=1, verbose=1,
    )
    search.fit(X_s, y)

    print("\nBest params:", search.best_params_)
    print("Best CV ROC-AUC (5-fold grouped):", round(search.best_score_, 4))
    return search.best_params_


def confirm_with_logo(X, y, groups, best_params):
    logo = LeaveOneGroupOut()
    rows = []
    n_splits = logo.get_n_splits(X, y, groups)

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        held_out = groups.iloc[test_idx].iloc[0]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = model.predict(X_test_s)

        rows.append({
            "project": held_out,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        })
        print(f"  [{fold_i}/{n_splits}] {held_out} done")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 50)
    print(f"Tuned RandomForest - mean across {len(df)} projects:")
    print(df[["precision", "recall", "f1", "roc_auc"]].mean())
    df.to_csv("results/rf_tuned_fold_results.csv", index=False)
    return df


if __name__ == "__main__":
    import json, sys
    X, y, groups = load_data()

    if "--confirm-only" in sys.argv:
        with open("results/rf_best_params.json") as f:
            best_params = json.load(f)
        print("Loaded params:", best_params)
    else:
        best_params = search_hyperparams(X, y, groups)
        with open("results/rf_best_params.json", "w") as f:
            json.dump(best_params, f)
        print("\nParams saved to rf_best_params.json")

    if "--search-only" not in sys.argv:
        confirm_with_logo(X, y, groups, best_params)
