import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
CHECKPOINT = "results/v4_origparams_results.csv"
with open("results/rf_best_params.json") as f:
    PARAMS = json.load(f)  # the ORIGINAL proven hyperparams, not v4's weak search

df = pd.read_csv("data/flakeflagger_engineered_v4.csv")
groups = df["project"]; y = df["flaky"]; X = df.drop(columns=["project", "flaky"])

logo = LeaveOneGroupOut()
try:
    done_df = pd.read_csv(CHECKPOINT)
    completed = set(done_df["project"].unique())
    print(f"Resuming: {len(completed)} done")
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
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **PARAMS)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    new_rows.append({
        "project": held_out,
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    })
    combined = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True) if done_df is not None else pd.DataFrame(new_rows)
    combined.to_csv(CHECKPOINT, index=False)
    print(f"  [{fold_i}/{n_splits}] {held_out} done - checkpointed")

df_final = pd.read_csv(CHECKPOINT)
if df_final["project"].nunique() >= groups.nunique():
    print("\n" + "="*50)
    print(df_final[["roc_auc","pr_auc","precision","recall","f1"]].mean())
else:
    print(f"[NOT COMPLETE] {df_final['project'].nunique()}/{groups.nunique()}")
