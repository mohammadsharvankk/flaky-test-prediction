"""
Feature engineering: raw FlakeFlagger data -> v2 (37 features) -> v3 (46 features)

INPUT:  FlakeFlagger_filtered_dataset.csv
        (source: uOttawa-Nanda-Lab/Flakify GitHub repo)
OUTPUT: flakeflagger_engineered.csv     (v2, 37 features)
        flakeflagger_engineered_v3.csv  (v3, 46 features)

NOTE: this is deterministic - no randomness. If you already have the two
output CSVs from earlier in the project, you do NOT need to rerun this -
they will be byte-for-byte identical either way. This script exists for
completeness/transparency and in case you need to regenerate from scratch.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/FlakeFlagger_filtered_dataset.csv"

SMELL_COLS = ['assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget',
              'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war']
CHURN_WINDOWS = ['hIndexModificationsPerCoveredLine_window5', 'hIndexModificationsPerCoveredLine_window10',
                  'hIndexModificationsPerCoveredLine_window25', 'hIndexModificationsPerCoveredLine_window50',
                  'hIndexModificationsPerCoveredLine_window75', 'hIndexModificationsPerCoveredLine_window100',
                  'hIndexModificationsPerCoveredLine_window500', 'hIndexModificationsPerCoveredLine_window10000']
WINDOW_SIZES = [5, 10, 25, 50, 75, 100, 500, 10000]


def build_v2(df):
    df = df.copy()
    missing = [c for c in SMELL_COLS + CHURN_WINDOWS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}\n"
                          "Check you downloaded the correct FlakeFlagger_filtered_dataset.csv.")

    df['smell_count'] = df[SMELL_COLS].sum(axis=1)
    df['has_any_smell'] = (df['smell_count'] > 0).astype(int)
    df['asserts_per_line'] = df['numAsserts'] / df['testLength'].replace(0, np.nan)
    df['asserts_per_line'] = df['asserts_per_line'].fillna(0)
    df['coverage_ratio'] = df['numCoveredLines'] / df['projectSourceLinesCovered'].replace(0, np.nan)
    df['coverage_ratio'] = df['coverage_ratio'].fillna(0)

    for col in ['testLength', 'numCoveredLines', 'ExecutionTime', 'projectSourceLinesCovered',
                'projectSourceClassesCovered', 'num_third_party_libs']:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    df['churn_short_to_long_ratio'] = (df[CHURN_WINDOWS[0]] + 1) / (df[CHURN_WINDOWS[-1]] + 1)
    df['churn_avg'] = df[CHURN_WINDOWS].mean(axis=1)

    return df


def build_v3(df_v2):
    df = df_v2.copy()

    df['execution_time_percentile_in_project'] = df.groupby('project')['ExecutionTime'].rank(pct=True)
    df['coverage_percentile_in_project'] = df.groupby('project')['coverage_ratio'].rank(pct=True)
    df['testlength_percentile_in_project'] = df.groupby('project')['testLength'].rank(pct=True)

    grp = df.groupby('project')['churn_avg']
    df['churn_zscore_in_project'] = (df['churn_avg'] - grp.transform('mean')) / grp.transform('std').replace(0, np.nan)
    df['churn_zscore_in_project'] = df['churn_zscore_in_project'].fillna(0)

    log_w = np.log(WINDOW_SIZES)
    mean_log_w = log_w.mean()
    denom = ((log_w - mean_log_w) ** 2).sum()

    def slope(row):
        y = row[CHURN_WINDOWS].values.astype(float)
        num = ((log_w - mean_log_w) * (y - y.mean())).sum()
        return num / denom if denom != 0 else 0.0

    df['churn_trend_slope'] = df.apply(slope, axis=1)
    df['execution_time_per_covered_line'] = df['ExecutionTime'] / (df['numCoveredLines'] + 1)
    df['smell_pair_assertroulette_mystery'] = df['assertion-roulette'] * df['mystery-guest']
    df['smell_pair_eager_conditional'] = df['eager-test'] * df['conditional-test-logic']
    df['smell_pair_resourceoptimism_indirect'] = df['resource-optimism'] * df['indirect-testing']

    return df


if __name__ == "__main__":
    print(f"Reading {RAW_PATH} ...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df_raw.shape}")

    df_v2 = build_v2(df_raw)
    df_v2.to_csv("data/flakeflagger_engineered.csv", index=False)
    print(f"Saved flakeflagger_engineered.csv - shape {df_v2.shape}")

    df_v3 = build_v3(df_v2)
    df_v3.to_csv("data/flakeflagger_engineered_v3.csv", index=False)
    print(f"Saved flakeflagger_engineered_v3.csv - shape {df_v3.shape}")
