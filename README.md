# Flaky Test Prediction with Explainable Machine Learning

**Predicting which tests in a CI/CD pipeline are likely to be flaky, and showing why a high score on a random split doesn't mean the model has learned anything.**

MSc Data Science Final Project, University of Birmingham Dubai (Distinction)
Mohammad Sharvan Kamil Karaya · Supervisor: Dr. Anis Zarrad

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange) ![SHAP](https://img.shields.io/badge/explainability-SHAP-purple) ![Dataset](https://img.shields.io/badge/data-FlakeFlagger%20%7C%20IDoFT-lightgrey)

**[Open the interactive dashboard →](https://mohammadsharvankk.github.io/flaky-test-prediction/)**

---

## The problem

Flaky tests pass or fail without any change to the code under test. In a large CI/CD setup they waste
debugging time, block releases and teach developers to ignore red builds. I dealt with this first-hand
over six years in software test engineering at Resideo (a Honeywell company), working across 500+ CI/CD
pipelines.

The usual fix is to rerun tests until they misbehave, which is slow and expensive. This project predicts
flakiness *ahead of time* from static and historical signals (execution time, coverage, code churn,
test smells), then explains every prediction, so a developer can see why a test was flagged.

## The main finding

| Evaluation split | ROC-AUC | PR-AUC |
|---|---|---|
| Random 5-fold (tests from the same project in train and test) | **0.971** | 0.770 |
| Leave-One-Project-Out (model never sees the target project) | **0.724** | 0.195 |

*Same model, same 46 features, same hyperparameters. Only the split changes.*
Reproduce with `python src/split_comparison.py` (about 1 minute).

Several published flaky-test models report 90%+ scores. This project shows that most of that gap comes
from **how the data is split, not how much the model learned**. With a random split, the model can learn
to recognise *projects* (their typical execution times, codebase size and churn patterns) and use that
to guess the label. Once a whole project is held out, which is the real situation when you point the
tool at a new codebase, performance falls to 0.72.

A second dataset makes the same point from another angle. On **IDoFT** (84.6% flaky), a Linear SVM
scores **F1 = 0.92 with ROC-AUC = 0.53**. It just predicts "flaky" for everything. F1 looks excellent
on a skewed dataset, even when the model can't tell a flaky test from a clean one.

## Approach

- **Data**: [FlakeFlagger](https://github.com/AlshammariA/FlakeFlagger) (Alshammari et al., ICSE 2021).
  21,661 tests from 23 open-source Java projects, labelled by rerunning each test 10,000 times. Only 3.7%
  are flaky.
- **Feature engineering** (`src/feature_engineering.py`): 24 base features → 37 (log transforms, ratios,
  churn slopes) → **46**, adding *project-relative* features such as a test's execution-time percentile
  within its own project. This was the biggest single improvement: ROC-AUC went from 0.700 to 0.724.
- **Models**: Logistic Regression, Linear SVM, Random Forest, Gradient Boosting, XGBoost, MLP, k-NN,
  and Voting/Stacking ensembles (two variants). **Random Forest won**. No ensemble beat it on ROC-AUC.
- **Evaluation**: Leave-One-Group-Out by project throughout: train on 22 projects, test on the 23rd.
- **Threshold tuning** (`src/tune_threshold.py`): the decision threshold is tuned with inner
  cross-validation on training projects only. With the model unchanged, mean F1 rose from **0.05 to 0.15**.
- **Explainability** (`src/shap_analysis_v3.py`): SHAP values for every test. The top drivers are
  execution time (raw, log, per covered line and project percentile), code churn on covered lines, and
  coverage size.

### Model comparison (37-feature set, Leave-One-Project-Out)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Random Forest (150 trees, untuned) | 0.702 | 0.185 | 0.019 |
| Stacking (5 models) | 0.701 | 0.195 | 0.159 |
| Random Forest (tuned) | 0.700 | 0.188 | 0.052 |
| Stacking (7 models, incl. MLP/k-NN) | 0.698 | 0.191 | 0.163 |
| Logistic Regression (baseline) | 0.627 | 0.162 | 0.134 |

With 3.7% flaky tests, a random guess scores a PR-AUC of about 0.04, so 0.19 is roughly 5× better than
chance. F1 values here use the default 0.5 threshold. Threshold tuning is what makes F1 usable.

## FlakeGuard dashboard

`dashboard/flakeguard_dashboard.html` is a single self-contained file. It needs no server: download it
and open it in a browser. Every number in it comes from the real out-of-fold results. It includes:

- Risk scores for all 21,661 tests, with a filter for each project
- Global and per-test SHAP explanations
- ROC curve and confusion matrix at the deployed threshold
- A **Predict** tab that runs all 300 trees of the trained Random Forest in the browser. You can score
  one test or upload a CSV (try `data/sample_mixed_tests.csv`).

## Repository layout

```
src/                        pipeline scripts (run from the repo root)
  feature_engineering.py      raw FlakeFlagger -> v2 (37) and v3 (46) feature sets
  run_models.py               7-model comparison, Leave-One-Project-Out
  diverse_ensemble.py         ensemble with MLP + k-NN added
  tune_random_forest.py       hyperparameter search + confirmation
  tune_threshold.py           per-fold threshold tuning with inner CV
  test_feature_v3.py          v2 vs v3 feature sets
  split_comparison.py         random split vs cross-project split (main finding)
  final_comparison.py         5-configuration results table
  compute_oof_scores_v3.py    out-of-fold risk score for every test
  shap_analysis_v3.py         SHAP values + final model
  prepare_dashboard_data.py   \
  export_predictor_model.py    > build the dashboard
  build_dashboard.py          /
  run_models_idoft_v2.py      IDoFT experiment
  *_v2*, test_v4_*            earlier phases, kept for the record (v4 features were rejected)
data/                       engineered feature sets + sample CSVs for the predictor
results/                    every CSV/JSON the scripts produce
models/final_model_v3.pkl   final Random Forest (scikit-learn 1.9.0)
dashboard/                  template + built dashboard
docs/                       RUNBOOK.md (full run order) and the project presentation
```

## Running it

```bash
git clone https://github.com/mohammadsharvankk/flaky-test-prediction.git
cd flaky-test-prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/split_comparison.py        # main finding, ~1 min
python src/compute_oof_scores_v3.py   # OOF risk scores (resumes from results/ if present)
python src/prepare_dashboard_data.py
python src/export_predictor_model.py
python src/build_dashboard.py         # -> dashboard/flakeguard_dashboard.html
```

Run every script from the repo root. [`docs/RUNBOOK.md`](docs/RUNBOOK.md) has the full pipeline in
order, with run times and notes on which scripts are checkpointed. The engineered datasets are already
in `data/`. To rebuild them from scratch, download `FlakeFlagger_filtered_dataset.csv` from the
[Flakify repo](https://github.com/uOttawa-Nanda-Lab/Flakify) into `data/raw/` and run
`python src/feature_engineering.py`.

Random Forest training varies slightly between runs and machines, so expect differences of about one
point of ROC-AUC.

## Limitations

- Results vary a lot between projects (cross-project ROC-AUC ranges from 0.15 to 1.00). Two projects
  score below 0.5, so a single model won't transfer to every codebase.
- Precision is still low at a usable recall. In practice this works best for ranking tests to
  investigate first, not as an automatic quarantine gate.
- All projects are Java. The features are language-neutral, but this hasn't been tested on other
  languages.

## AI assistance

Parts of this project were built with an AI coding assistant, within the assignment's 20% allowance:
mainly boilerplate code, debugging pipeline errors and drafting the dashboard front end. The feature
engineering decisions, the evaluation methodology, the decision to reject the v4 feature experiment
after it proved inconsistent across environments, and the interpretation of the results are my own.

## References

- Alshammari, A., Morris, C., Hilton, M., Bell, J. *FlakeFlagger: Predicting Flakiness Without Rerunning Tests.* ICSE 2021.
- Fatima, S., Ghaleb, T. A., Briand, L. *Flakify: A Black-Box, Language Model-Based Predictor for Flaky Tests.* IEEE TSE 2022.
- Lam, W. et al. *International Dataset of Flaky Tests (IDoFT).*
- Lundberg, S., Lee, S.-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The FlakeFlagger and IDoFT data remain under
their original authors' terms. The files in `data/` are derived from them for research use.
