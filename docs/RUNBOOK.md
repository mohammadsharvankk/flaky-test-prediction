# Full Local Rerun — Master Runbook

Run every command from the **repo root**, inside a virtual environment with
`requirements.txt` installed. Run each command in order. Where a script is "checkpointed," it's safe to stop and
re-run the same command later — it picks up where it left off.

---

## Part 0 — Files you need in place first

These are already in the repo under `data/`:

- `flakeflagger_engineered.csv` (v2, 37 features)
- `flakeflagger_engineered_v3.csv` (v3, 46 features)

**If you've lost either of these** and need to regenerate from scratch, you'll also need
the raw `FlakeFlagger_filtered_dataset.csv` (place it in `data/raw/`) (source: the `uOttawa-Nanda-Lab/Flakify`
GitHub repo), then run:

```bash
python src/feature_engineering.py
```

This is deterministic — no randomness — so re-running it (or not) makes no difference
to your final numbers. Skip this step entirely if you still have the two CSVs.

---

## Part 1 — Core model comparison

```bash
python src/run_models.py
```
Outputs: `fold_results.csv`, `model_summary.csv` — the 7-model comparison (base models
+ Voting + Stacking) that feeds the deck's Model Comparison chart. **Not checkpointed —
let it finish in one sitting** (~5-10 min depending on your machine).

```bash
python src/diverse_ensemble.py
```
Outputs: `diverse_ensemble_results.csv`, `diverse_ensemble_summary.csv` — the follow-up
test with a genuinely diverse ensemble (neural net + k-NN included).

---

## Part 2 — Hyperparameter and threshold tuning

```bash
python src/tune_random_forest.py --search-only
python src/tune_random_forest.py --confirm-only
```
Outputs: `rf_best_params.json`, `rf_tuned_fold_results.csv`. The confirm step is
checkpointed — if it stops partway, just re-run the same command.

```bash
python src/tune_threshold.py
```
Outputs: `threshold_tuned_results.csv` — the per-fold threshold tuning result.

---

## Part 3 — v3 feature validation

```bash
python src/test_feature_v3.py
```
Outputs: `feature_v3_test_results.csv`, `feature_v3_test_summary.csv` — confirms
whether the project-relative features (v3) genuinely beat the original set (v2) on
your machine.

*(Optional, since it was already rejected — only rerun if you want to double check):*
```bash
python src/test_v4_with_orig_params.py
```

---

## Part 3b — Random split vs cross-project split

```bash
python src/split_comparison.py
```
Outputs: `results/split_comparison.csv` — the same tuned Random Forest on the v3
features, evaluated with a shuffled stratified 5-fold split and with
Leave-One-Project-Out. About 1 minute. This is the evidence for the main finding.

---

## Part 4 — Final comparison table

```bash
python src/final_comparison.py
```
Outputs: `final_comparison_results.csv`, `final_comparison_summary.csv` — the 5-config
table used in the deck's Results slide (ROC-AUC, PR-AUC, F1, cost for each).
**Not checkpointed — this is the longest single run (~20-30 min)**, since it includes
Stacking. Let it finish uninterrupted.

---

## Part 5 — SHAP + OOF scores for the ACTUAL deployed model (v3)

This is the model your dashboard and deck's Related Work slide represent — make sure
this part completes even if you skip everything else.

```bash
python src/compute_oof_scores_v3.py
```
Outputs: `oof_scores_v3.csv`, `global_threshold_v3.json`. Checkpointed — safe to
re-run if interrupted.

```bash
python src/shap_analysis_v3.py
```
Outputs: `final_model_v3.pkl`, `shap_feature_importance_v3.csv`,
`shap_values_per_test_v3.csv`. Not checkpointed, but fast (~2 min).

---

## Part 6 — Build the dashboard from YOUR numbers

Three scripts turn the results into the dashboard:

```bash
python src/prepare_dashboard_data.py
```
Reads your `oof_scores_v3.csv`, `shap_feature_importance_v3.csv`,
`shap_values_per_test_v3.csv`, `global_threshold_v3.json` → produces
`dashboard_data.json`. Prints your confirmed ROC-AUC at the end — that's your real
number, straight from this run.

```bash
python src/export_predictor_model.py
```
Reads `final_model_v3.pkl` + `flakeflagger_engineered_v3.csv` → exports all 300 trees
to `predictor_trees.json` (~7.5 MB) plus `predictor_ui.json`. Takes 1-2 minutes.

```bash
python src/build_dashboard.py
```
Combines `dashboard_template.html` + the three JSON files above →
`flakeguard_dashboard.html`. This is your final, complete, self-contained dashboard —
built entirely from numbers your own machine produced.

**Open it and click through every tab** to confirm it looks right before using it for
the demo.

---

## Quick reference — what depends on what

```
raw data ──> feature_engineering.py ──> flakeflagger_engineered_v3.csv
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                          ▼                         ▼
             run_models.py          compute_oof_scores_v3.py    tune_random_forest.py
             final_comparison.py    shap_analysis_v3.py         tune_threshold.py
                    │                          │
                    │                          ▼
                    │                 prepare_dashboard_data.py
                    │                 export_predictor_model.py
                    │                          │
                    │                          ▼
                    │                  build_dashboard.py
                    │                          │
                    ▼                          ▼
          deck numbers (manual)      flakeguard_dashboard.html
```

Parts 1-4 feed the numbers you'd manually put in your deck (I can help rebuild the
deck from your fresh numbers once you have them). Parts 5-6 feed the dashboard
directly and automatically — no manual number-copying needed there.

---

## Where files live

| Folder | Contents |
|---|---|
| `data/` | engineered feature sets and sample CSVs for the dashboard's batch predictor |
| `data/raw/` | put the raw FlakeFlagger CSV here if you regenerate features (not committed) |
| `results/` | every CSV/JSON a script writes: fold results, summaries, OOF scores, SHAP values, tuned params |
| `models/` | `final_model_v3.pkl` (scikit-learn 1.9.0) |
| `dashboard/` | template + the built `flakeguard_dashboard.html`; intermediates go to `dashboard/build/` (not committed) |
