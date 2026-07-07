# Phase 3 — XGBoost Model Training

**Project:** AI-Powered Heatstroke Early Warning System for Outdoor Workers  
**Input:** `workers_with_clusters.csv` (Phase 2 output — 30 columns, 5,450 workers)  
**Outputs:** Trained model + 9 diagnostic plots + 4 model artefact files

---

## What Phase 3 Does and Why

Phase 3 is where unsupervised learning (Phase 2) meets supervised learning. The cluster persona each worker was assigned in Phase 2 becomes a feature (`cluster_id`) that tells XGBoost "this is the type of person you are predicting for." Combined with 26 other features including lag columns and derived heat-stress metrics, XGBoost learns the complex, non-linear mapping from physiological and environmental state to risk class.

The core output is `heatstroke_model.pkl` — a trained classifier that takes a 27-feature vector describing a worker's current and recent state, and returns a probability distribution across `[Low, Moderate, High, Critical]`. The app fires an alert when `P(High) + P(Critical)` exceeds the worker's adaptive threshold.

---

## Phase 1 Mathematical Upgrades — How Phase 3 Handles All Three

| Phase 1 Upgrade | Phase 3 Handling |
|---|---|
| **Predictive target shift (t+30):** Risk labels reflect future state. | Pass `--target risk_label_future_num` to `run_phase3.py`. The entire pipeline is target-agnostic — only the `TARGET_COL` constant changes. The model then learns: "given this worker's readings now, what class will they be in 30 minutes?" |
| **Non-linear interaction:** Exponential temp×humidity synergy term. | `heat_index` and `temp_humidity_product` are both in the feature matrix. These pre-computed non-linear terms expose the synergistic penalty directly to XGBoost, which then learns to weight them via gradient boosting. SHAP confirms `heat_index` ranks in the top 5. |
| **Dynamic lag physiology:** HR lag driven by `work_hours`, `acclimatisation_days`, `metabolic_rate`. | `hr_delta_t15` and `hr_delta_t30` (the dynamic trajectory outputs) are in the feature matrix. The model learns that a rising HR trajectory over 30 minutes is a stronger danger signal than any single-point reading. SHAP confirms `hr_delta_t30` ranks top-10 globally. |

---

## Why XGBoost + Lag Features Instead of LSTM

This is the most common viva question about Phase 3. The complete answer:

**Time-series capability without sequential data.** By including `heart_rate_t15`, `heart_rate_t30`, `ambient_temp_t15`, `ambient_temp_t30`, `hr_delta_t15`, `hr_delta_t30` as explicit columns, XGBoost sees the trajectory of each measurement across three time points. It learns that a worker whose HR rose 16 bpm in 30 minutes under 44°C is heading toward collapse — the same insight an LSTM would extract, without requiring a sequential architecture.

**Practical superiority for this dataset size and deployment target:**

| Criterion | XGBoost + Lag | LSTM |
|---|---|---|
| Dataset size requirement | Works well < 10K samples | Needs >> 10K sequences |
| Training time | Seconds to minutes | Minutes to hours |
| Inference time | < 1ms per worker | 10–100ms per sequence |
| Smartphone deployment | `.pkl` or ONNX, trivial | TensorFlow Lite, complex |
| Explainability | SHAP values, exact | Attention weights, approximate |
| Missing data handling | Native (split on NaN) | Requires imputation |
| Hyperparameter sensitivity | Moderate | High (architecture choice) |

---

## Feature Matrix (27 Features)

| Group | Features | Count |
|---|---|---|
| Personal | `age`, `bmi`, `acclimatisation_days` | 3 |
| Environmental (current) | `ambient_temp`, `humidity`, `wind_speed`, `solar_radiation` | 4 |
| Environmental (lag) | `ambient_temp_t15`, `ambient_temp_t30`, `humidity_t15`, `humidity_t30` | 4 |
| Work | `metabolic_rate`, `work_hours`, `hydration_level` | 3 |
| Physiological | `heart_rate`, `heart_rate_t15`, `heart_rate_t30`, `sweat_rate` | 4 |
| PHS output | `core_temp_tre` | 1 |
| Derived | `heat_index`, `hr_delta_t15`, `hr_delta_t30`, `temp_delta_t15`, `temp_humidity_product` | 5 |
| Phase 2 cluster | `cluster_id`, `vulnerability_score`, `adaptive_alert_multiplier` | 3 |

**Target:** `risk_label_num` (0=Low, 1=Moderate, 2=High, 3=Critical)  
**Target if t+30 upgrade applied:** `risk_label_future_num`

---

## Evaluation Strategy — Why Not Accuracy

In safety-critical AI, evaluation metric choice is a design decision, not a technicality.

**Sensitivity (Recall) for High + Critical — PRIMARY metric**  
A false negative (missing a worker who is about to collapse) is categorically different from a false positive (telling someone to rest unnecessarily). Recall directly measures how many true danger cases the model catches. We set hard constraints: `Recall(High) ≥ 0.75` and `Recall(Critical) ≥ 0.80`. A model failing either constraint is rejected during tuning, regardless of its F1 score.

**F1-macro — SECONDARY metric**  
F1 balances precision and recall across all four classes. Macro-averaging gives equal weight to minority classes (Low, Moderate), preventing the model from ignoring them entirely in favour of the dominant Critical class.

**ROC-AUC (one-vs-rest per class) — TERTIARY metric**  
Shows threshold-independent discriminative ability. An AUC near 1.0 means the model can reliably separate that class from all others regardless of which probability threshold is chosen.

**Accuracy — REPORTED BUT NOT HEADLINE**  
On our dataset (60.7% Critical), a model that predicts Critical for every worker achieves 60.7% accuracy. This is meaningless. Accuracy is reported for completeness, but never used for model selection or as a presentation headline.

---

## Actual Pipeline Results (from executed run)

```
PHASE 3 — XGBOOST TRAINING — SUMMARY REPORT
================================================================
Prediction target      : risk_label_num
Features used          : 27
SMOTE applied          : True
Tuning method          : optuna (30 trials)

BEST HYPERPARAMETERS
  n_estimators        : 596
  max_depth           : 3
  learning_rate       : 0.271
  subsample           : 0.933
  colsample_bytree    : 0.606
  min_child_weight    : 2
  gamma               : 0.092
  reg_alpha           : 0.304
  reg_lambda          : 1.812

5-FOLD CROSS-VALIDATION
  F1-macro        : 0.9901 ± 0.0029
  Recall (High)   : 0.9936 ± 0.0052
  Recall (Critical): 0.9965 ± 0.0035

TEST SET — HEADLINE METRICS
  F1-macro        : 0.9772  ← primary metric
  ROC-AUC macro   : 0.9999
  Accuracy        : 0.9902  (not headline)

PER-CLASS METRICS
  Class      Sensitivity      F1     AUC   Precision     N
  Low          1.0000      0.9756  1.0000   0.9524      60
  Moderate     0.9000      0.9474  1.0000   1.0000      60
  High         1.0000      0.9878  0.9998   0.9758     202
  Critical     0.9960      0.9980  0.9999   1.0000     496

SAFETY CONSTRAINT STATUS
  Recall(High)     ≥ 0.75 : ✓ PASS  (actual = 1.0000)
  Recall(Critical) ≥ 0.80 : ✓ PASS  (actual = 0.9960)

CONFUSION MATRIX
             Low  Moderate    High  Critical
  Low         60         0       0         0
  Moderate     3        54       3         0
  High         0         0     202         0
  Critical     0         0       2       494
```

**What these results mean:** The model misses zero High-risk workers and misses only 2 Critical workers (out of 496 in the test set). The 3 Moderate workers misclassified as Low represent a false negative scenario for the lowest danger class — still worth noting but not a safety failure.

---

## SHAP Top 10 Features (from executed run)

```
core_temp_tre            : 2.5440  (PHS-predicted core temperature — dominant)
acclimatisation_days     : 0.3257  (protective factor — highly influential)
cluster_id               : 0.2508  (Phase 2 persona — 3rd most important feature)
vulnerability_score      : 0.1613  (composite risk score from Phase 2)
heat_index               : 0.1339  (non-linear temp×humidity interaction)
temp_humidity_product    : 0.1262  (Phase 1 synergy term)
work_hours               : 0.1057  (cumulative heat load over shift)
wind_speed               : 0.0922  (cooling effect)
solar_radiation          : 0.0661  (radiant heat load)
hr_delta_t30             : 0.0513  (dynamic HR trajectory — Phase 1 upgrade)
```

**Key insight for viva:** `cluster_id` is the 3rd most important feature globally. This directly validates that Phase 2 (K-Means profiling) made a meaningful contribution to the model. Without Phase 2, this feature would not exist and the model would be less accurate and less personalised.

---

## File Structure

```
heatstroke_ai/
├── model/
│   ├── train_xgboost.py         ← core training engine
│   ├── shap_interpreter.py      ← SHAP computation and 4 SHAP plots
│   ├── visualise_phase3.py      ← 5 evaluation plots
│   └── run_phase3.py            ← entry point — run this
├── models/
│   ├── heatstroke_model.pkl     ← trained XGBoost (for app)
│   ├── feature_list.json        ← exact feature order for inference
│   ├── best_params.json         ← Optuna-selected hyperparameters
│   └── phase3_metrics.json      ← all evaluation metrics (JSON)
├── data/
│   └── phase3_summary_report.txt
└── outputs/
    └── phase3_plots/
        ├── plot1_confusion_matrix.png
        ├── plot2_roc_curves.png
        ├── plot3_learning_curves.png
        ├── plot4_class_probabilities.png
        ├── plot5_demo_scenarios.png
        ├── shap1_global_importance.png
        ├── shap2_per_class_importance.png
        ├── shap3_dependency_plots.png
        └── shap4_waterfall_demo.png
```

---

## Dependencies

```bash
pip install xgboost scikit-learn imbalanced-learn shap optuna \
            matplotlib seaborn joblib pandas numpy
```

---

## How to Run

```bash
# Standard run — Optuna tuning, SMOTE, current risk target
python run_phase3.py

# Full options
python run_phase3.py \
  --input    ../data/workers_with_clusters.csv \
  --models_dir ../models \
  --plots_dir  ../outputs/phase3_plots \
  --search   optuna \
  --trials   40 \
  --seed     42

# Phase 1 t+30 upgrade (predict future risk)
python run_phase3.py --target risk_label_future_num

# Quick run — skip Optuna, use preset params
python run_phase3.py --search default --no_shap

# No SMOTE — use class weights instead
python run_phase3.py --no_smote
```

---

## File 1 of 4 — `train_xgboost.py`

Core training engine. All functions and configuration constants.

```python
"""
=============================================================================
PHASE 3: XGBOOST MODEL TRAINING — HEATSTROKE EARLY WARNING SYSTEM
=============================================================================
Project : AI-Powered Heatstroke Early Warning System for Outdoor Workers
Input   : workers_with_clusters.csv  (Phase 2 output — 30 columns)
Output  : heatstroke_model.pkl, feature_list.json, best_params.json,
          phase3_metrics.json, phase3_summary_report.txt

Prediction Target:
    risk_label_num (0=Low, 1=Moderate, 2=High, 3=Critical)
    Set TARGET_COL = "risk_label_future_num" for Phase 1 t+30 upgrade.

Why XGBoost + Lag Features:
    Lag features (t-15, t-30) give XGBoost time-series trajectory awareness
    without LSTM complexity. Runs in ms on smartphones. SHAP-explainable.
    Outperforms LSTM on tabular lag datasets under 10K samples.

Evaluation Strategy:
    PRIMARY  : Sensitivity (Recall) for High + Critical classes
    SECONDARY: F1-macro
    TERTIARY : ROC-AUC one-vs-rest
    NOT LEAD : Accuracy
=============================================================================
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, roc_auc_score, accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TARGET_COL = "risk_label_num"
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
N_CLASSES   = 4

EXCLUDE_COLS = [
    "risk_label_str", "risk_label_num", "risk_label_future_num",
    "persona_name", TARGET_COL,
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

USE_SMOTE     = True
CV_FOLDS      = 5
SEARCH_METHOD = "optuna"
OPTUNA_TRIALS = 40

DEFAULT_PARAMS = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.80, "colsample_bytree": 0.80,
    "min_child_weight": 3, "gamma": 0.10,
    "reg_alpha": 0.10, "reg_lambda": 1.50,
}

MIN_HIGH_RECALL     = 0.75
MIN_CRITICAL_RECALL = 0.80
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> tuple:
    """
    Load Phase 2 dataset. Build feature matrix X (27 cols) and target y.

    Feature groups:
        Personal      : age, bmi, acclimatisation_days
        Environmental : ambient_temp, humidity, wind_speed, solar_radiation
        Env Lags      : ambient_temp_t15, ambient_temp_t30,
                        humidity_t15, humidity_t30
        Work          : metabolic_rate, work_hours, hydration_level
        Physiological : heart_rate, heart_rate_t15, heart_rate_t30, sweat_rate
        PHS Output    : core_temp_tre
        Derived       : heat_index, hr_delta_t15, hr_delta_t30,
                        temp_delta_t15, temp_humidity_product
        Phase 2       : cluster_id, vulnerability_score,
                        adaptive_alert_multiplier
    """
    df = pd.read_csv(csv_path)
    exclude_set = set(EXCLUDE_COLS)
    feature_cols = [
        c for c in df.columns
        if c not in exclude_set and df[c].dtype != object and c != TARGET_COL
    ]
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not in dataset. "
                         f"Set TARGET_COL = 'risk_label_future_num' for t+30 upgrade.")
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values.astype(int)

    print(f"   Features      : {len(feature_cols)} columns")
    print(f"   Samples       : {len(X):,}")
    print(f"   Target column : {TARGET_COL}")
    for cls, label in enumerate(RISK_LABELS):
        n = (y == cls).sum()
        print(f"      Class {cls} ({label:10s}): {n:5d} ({n/len(y)*100:.1f}%)")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(X, y):
    """
    Stratified 70/15/15 split preserving class proportions.
    Two-step: first carve test, then split train/val from remainder.
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=SEED)
    val_of_tv = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_of_tv, stratify=y_tv, random_state=SEED)

    print(f"\n   Split sizes:")
    print(f"      Train : {len(X_train):5,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"      Val   : {len(X_val):5,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"      Test  : {len(X_test):5,} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_class_imbalance(X_train, y_train):
    """
    Apply SMOTE to training set ONLY.

    SMOTE generates synthetic minority-class samples by interpolating
    between k-nearest neighbours. Applied only to training data — test and
    validation sets remain the original unmodified distribution.

    Returns numpy arrays (X_res, y_res).
    """
    if not USE_SMOTE:
        print("   SMOTE disabled — using class_weight in XGBoost instead.")
        return X_train.values, y_train

    print("   Applying SMOTE to training set...")
    for cls, label in enumerate(RISK_LABELS):
        print(f"      Before {label:10s}: {(y_train==cls).sum():5d}")

    smote = SMOTE(sampling_strategy="not majority", k_neighbors=3, random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train.values, y_train)

    for cls, label in enumerate(RISK_LABELS):
        print(f"      After  {label:10s}: {(y_res==cls).sum():5d}")

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING — OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

def tune_with_optuna(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    """
    Bayesian hyperparameter optimisation using Optuna TPE sampler.

    Objective: Maximise F1-macro on validation set subject to safety constraints:
        Recall(High)     >= MIN_HIGH_RECALL
        Recall(Critical) >= MIN_CRITICAL_RECALL
    Trials violating either constraint receive score = 0.0 (rejected).

    Search space:
        n_estimators    : [100, 800]       — boosting rounds
        max_depth       : [3, 10]          — tree complexity
        learning_rate   : [0.01, 0.30]     — shrinkage (log-uniform)
        subsample       : [0.60, 1.00]     — row sampling
        colsample_bytree: [0.50, 1.00]     — column sampling
        min_child_weight: [1, 10]          — leaf regularisation
        gamma           : [0.00, 0.50]     — min split gain
        reg_alpha       : [0.00, 1.00]     — L1 regularisation
        reg_lambda      : [0.50, 3.00]     — L2 regularisation
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "random_state": SEED, "verbosity": 0, "n_jobs": -1,
            "early_stopping_rounds": 30,
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample":        trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.00, 0.50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.00, 1.00),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.50, 3.00),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        recalls = recall_score(y_val, y_pred, average=None, zero_division=0)
        if (len(recalls) > 3
                and recalls[2] < MIN_HIGH_RECALL
                or (len(recalls) > 3 and recalls[3] < MIN_CRITICAL_RECALL)):
            return 0.0  # safety constraint violated — reject trial

        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n   Optuna completed {n_trials} trials")
    print(f"   Best F1-macro on validation: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    return study.best_params


def tune_with_grid(X_train, y_train, X_val, y_val):
    """
    Reduced grid search fallback (16 combinations).
    Used when Optuna is unavailable or for quick test runs.
    """
    from itertools import product as iterproduct
    param_grid = {
        "n_estimators": [200, 400], "max_depth": [5, 7],
        "learning_rate": [0.05, 0.10], "subsample": [0.80],
        "colsample_bytree": [0.80], "min_child_weight": [3],
        "gamma": [0.10], "reg_alpha": [0.10], "reg_lambda": [1.50],
    }
    keys, values = list(param_grid.keys()), list(param_grid.values())
    combos = list(iterproduct(*values))
    best_f1, best_params = 0.0, {}

    for combo in combos:
        params = dict(zip(keys, combo))
        params.update({"objective": "multi:softprob", "num_class": N_CLASSES,
                       "tree_method": "hist", "random_state": SEED,
                       "verbosity": 0, "n_jobs": -1})
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = m.predict(X_val)
        recalls = recall_score(y_val, y_pred, average=None, zero_division=0)
        if len(recalls) > 3 and recalls[2] >= MIN_HIGH_RECALL and recalls[3] >= MIN_CRITICAL_RECALL:
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_params = f1, dict(zip(keys, combo))

    if not best_params:
        best_params = dict(zip(keys, combos[0]))
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train final XGBoost with best hyperparameters.

    Key settings:
        early_stopping_rounds=50  — stops if val loss doesn't improve for 50 rounds
        eval_metric = mlogloss    — multi-class log-loss (more sensitive than merror)
        tree_method = hist        — histogram-based splits (fast, memory-efficient)

    Note: early_stopping_rounds is in the CONSTRUCTOR in XGBoost ≥ 2.0,
    not in .fit(). This code targets XGBoost 2.x / 3.x.
    """
    params = {
        "objective": "multi:softprob", "num_class": N_CLASSES,
        "eval_metric": ["mlogloss", "merror"],
        "tree_method": "hist", "random_state": SEED,
        "verbosity": 0, "n_jobs": -1,
        "early_stopping_rounds": 50,
        **best_params,
    }
    if params.get("n_estimators", 300) < 400:
        params["n_estimators"] = 400

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=False)

    print(f"\n   Training complete.")
    print(f"   Best iteration  : {model.best_iteration}")
    try:
        r = model.evals_result()
        print(f"   Final train log-loss : {r['validation_0']['mlogloss'][model.best_iteration]:.4f}")
        print(f"   Final val   log-loss : {r['validation_1']['mlogloss'][model.best_iteration]:.4f}")
    except Exception:
        pass

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(X_train_full, y_train_full, best_params, n_folds=CV_FOLDS):
    """
    5-fold stratified cross-validation on full training data.
    SMOTE is applied inside each fold to prevent data leakage.
    Reports F1-macro, Recall(High), Recall(Critical) as mean ± std.
    """
    params = {"objective": "multi:softprob", "num_class": N_CLASSES,
              "tree_method": "hist", "random_state": SEED,
              "verbosity": 0, "n_jobs": -1, **best_params}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    f1_scores, high_recalls, critical_recalls = [], [], []
    X_arr = X_train_full.values if hasattr(X_train_full, "values") else X_train_full

    for tr_idx, vl_idx in skf.split(X_arr, y_train_full):
        X_tr, X_vl = X_arr[tr_idx], X_arr[vl_idx]
        y_tr, y_vl = y_train_full[tr_idx], y_train_full[vl_idx]

        if USE_SMOTE:
            smote = SMOTE(sampling_strategy="not majority", k_neighbors=3, random_state=SEED)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        y_pred = m.predict(X_vl)

        f1_scores.append(f1_score(y_vl, y_pred, average="macro", zero_division=0))
        recalls = recall_score(y_vl, y_pred, average=None, zero_division=0)
        high_recalls.append(recalls[2] if len(recalls) > 2 else 0)
        critical_recalls.append(recalls[3] if len(recalls) > 3 else 0)

    cv = {
        "f1_macro_mean":        float(np.mean(f1_scores)),
        "f1_macro_std":         float(np.std(f1_scores)),
        "high_recall_mean":     float(np.mean(high_recalls)),
        "high_recall_std":      float(np.std(high_recalls)),
        "critical_recall_mean": float(np.mean(critical_recalls)),
        "critical_recall_std":  float(np.std(critical_recalls)),
    }
    print(f"\n   {n_folds}-Fold Cross-Validation Results:")
    print(f"   F1-macro        : {cv['f1_macro_mean']:.4f} ± {cv['f1_macro_std']:.4f}")
    print(f"   Recall (High)   : {cv['high_recall_mean']:.4f} ± {cv['high_recall_std']:.4f}")
    print(f"   Recall (Critical): {cv['critical_recall_mean']:.4f} ± {cv['critical_recall_std']:.4f}")
    return cv


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, X_test, y_test, feature_names):
    """
    Final unbiased evaluation on held-out test set.
    Computes per-class and aggregate metrics. Checks safety constraints.
    """
    X_arr  = X_test.values if hasattr(X_test, "values") else X_test
    y_pred = model.predict(X_arr)
    y_prob = model.predict_proba(X_arr)

    report   = classification_report(y_test, y_pred, target_names=RISK_LABELS,
                                     output_dict=True, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)
    recalls  = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per   = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc      = accuracy_score(y_test, y_pred)

    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc_per, auc_macro = [], 0.0
    try:
        for i in range(N_CLASSES):
            auc_per.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        auc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc_per = [0.0] * N_CLASSES

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted",
                                             zero_division=0)), 4),
        "roc_auc_macro": round(float(auc_macro), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }
    for i, label in enumerate(RISK_LABELS):
        metrics["per_class"][label] = {
            "sensitivity_recall": round(float(recalls[i]), 4),
            "f1_score":           round(float(f1_per[i]), 4),
            "roc_auc":            round(float(auc_per[i]) if i < len(auc_per) else 0.0, 4),
            "precision":          round(float(report[label]["precision"]), 4),
            "support":            int(report[label]["support"]),
        }

    print(f"\n   TEST SET RESULTS")
    print(f"   F1-macro       : {f1_macro:.4f}")
    print(f"   ROC-AUC macro  : {auc_macro:.4f}")
    print(f"   Accuracy       : {acc:.4f}  (not headline)")
    print(f"\n   {'Class':<12} {'Recall':>10} {'F1':>8} {'AUC':>8}")
    for i, label in enumerate(RISK_LABELS):
        print(f"   {label:<12} {recalls[i]:>10.4f} {f1_per[i]:>8.4f} "
              f"{auc_per[i] if i < len(auc_per) else 0.0:>8.4f}")

    h_ok = recalls[2] >= MIN_HIGH_RECALL
    c_ok = recalls[3] >= MIN_CRITICAL_RECALL
    print(f"\n   SAFETY CHECK:")
    print(f"   Recall(High)     = {recalls[2]:.4f}  ({'✓ PASS' if h_ok else '✗ FAIL'})")
    print(f"   Recall(Critical) = {recalls[3]:.4f}  ({'✓ PASS' if c_ok else '✗ FAIL'})")

    return metrics, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_model_artefacts(model, feature_names, best_params, metrics, cv_results, models_dir):
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, Path(models_dir) / "heatstroke_model.pkl")
    print(f"  ✓ Model saved          : {models_dir}/heatstroke_model.pkl")

    with open(Path(models_dir) / "feature_list.json", "w") as f:
        json.dump({"features": feature_names, "target": TARGET_COL,
                   "n_classes": N_CLASSES, "risk_labels": RISK_LABELS}, f, indent=2)
    print(f"  ✓ Feature list saved   : {models_dir}/feature_list.json")

    with open(Path(models_dir) / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  ✓ Best params saved    : {models_dir}/best_params.json")

    with open(Path(models_dir) / "phase3_metrics.json", "w") as f:
        json.dump({**metrics, "cross_validation": cv_results}, f, indent=2)
    print(f"  ✓ Metrics saved        : {models_dir}/phase3_metrics.json")


def export_phase3_report(metrics, cv_results, best_params, feature_names, elapsed, output_dir):
    cm = np.array(metrics["confusion_matrix"])
    lines = [
        "=" * 65, "PHASE 3 — XGBOOST TRAINING — SUMMARY REPORT", "=" * 65,
        f"Prediction target      : {TARGET_COL}",
        f"Features used          : {len(feature_names)}",
        f"SMOTE applied          : {USE_SMOTE}",
        f"Tuning method          : {SEARCH_METHOD}",
        f"Elapsed time           : {elapsed:.1f}s", "",
        "BEST HYPERPARAMETERS", "-" * 50,
    ]
    for k, v in best_params.items():
        lines.append(f"  {k:<25}: {v}")
    lines += [
        "", f"CROSS-VALIDATION ({CV_FOLDS}-FOLD STRATIFIED)", "-" * 50,
        f"  F1-macro        : {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}",
        f"  Recall (High)   : {cv_results['high_recall_mean']:.4f} ± {cv_results['high_recall_std']:.4f}",
        f"  Recall (Critical): {cv_results['critical_recall_mean']:.4f} ± {cv_results['critical_recall_std']:.4f}",
        "", "TEST SET — HEADLINE METRICS", "-" * 50,
        f"  F1-macro        : {metrics['f1_macro']:.4f}  ← primary metric",
        f"  ROC-AUC macro   : {metrics['roc_auc_macro']:.4f}",
        f"  Accuracy        : {metrics['accuracy']:.4f}  (not headline)",
        "", "TEST SET — PER-CLASS METRICS", "-" * 50,
        f"  {'Class':<12} {'Sensitivity':>12} {'F1':>8} {'AUC':>8} {'Precision':>10} {'N':>6}",
        "  " + "-" * 58,
    ]
    for label in RISK_LABELS:
        pc = metrics["per_class"][label]
        lines.append(f"  {label:<12} {pc['sensitivity_recall']:>12.4f} {pc['f1_score']:>8.4f} "
                     f"{pc['roc_auc']:>8.4f} {pc['precision']:>10.4f} {pc['support']:>6}")

    lines += ["", "CONFUSION MATRIX", "-" * 50,
              f"  Rows=Actual  Cols=Predicted  Labels={RISK_LABELS}", ""]
    header = "  " + " ".join(f"{l[:4]:>8}" for l in RISK_LABELS)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append("  " + f"{RISK_LABELS[i][:4]:<6}" + " ".join(f"{v:>8}" for v in row))

    lines += ["", "SAFETY CONSTRAINT STATUS", "-" * 50,
              f"  Recall(High)     >= {MIN_HIGH_RECALL} : "
              f"{'PASS' if metrics['per_class']['High']['sensitivity_recall'] >= MIN_HIGH_RECALL else 'FAIL'}",
              f"  Recall(Critical) >= {MIN_CRITICAL_RECALL} : "
              f"{'PASS' if metrics['per_class']['Critical']['sensitivity_recall'] >= MIN_CRITICAL_RECALL else 'FAIL'}",
              "", "FEATURES USED", "-" * 50]
    for feat in feature_names:
        lines.append(f"  {feat}")

    path = Path(output_dir) / "phase3_summary_report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Summary report saved : {path}")
```

---

## File 2 of 4 — `shap_interpreter.py`

SHAP computation and 4 explainability plots.

```python
"""
=============================================================================
PHASE 3 — SHAP INTERPRETABILITY MODULE
=============================================================================
Uses TreeExplainer (exact, fast, native XGBoost support).
Generates 4 plots:
  shap1_global_importance.png   — mean |SHAP| bar chart
  shap2_per_class_importance.png— per-class top features
  shap3_dependency_plots.png    — SHAP vs feature value for top 3 features
  shap4_waterfall_demo.png      — why did the model predict this worker?
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from pathlib import Path

RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS = {"Low": "#27AE60", "Moderate": "#F39C12",
               "High": "#E67E22", "Critical": "#C0392B"}


def compute_shap_values(model, X_test, feature_names):
    """Compute SHAP values using TreeExplainer. Returns (explainer, shap_stack, shap_df)."""
    explainer = shap.TreeExplainer(model)
    X_arr     = X_test.values if hasattr(X_test, "values") else X_test
    shap_vals = explainer.shap_values(X_arr)

    if isinstance(shap_vals, list):
        shap_stack = np.stack(shap_vals, axis=-1)
    else:
        shap_stack = shap_vals

    mean_abs  = np.mean(np.abs(shap_stack).mean(axis=-1), axis=0)
    shap_df   = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n   SHAP Top 10 Features:")
    for _, row in shap_df.head(10).iterrows():
        print(f"      {row['feature']:28s}: {row['mean_abs_shap']:.4f}")

    return explainer, shap_stack, shap_df


def plot_shap_summary_bar(shap_df, plots_dir, top_n=15):
    """Global feature importance bar chart — primary explainability slide."""
    top    = shap_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#C0392B" if "heart_rate" in f or "hr_delta" in f
              else "#2980B9" if "temp" in f or "humidity" in f
              else "#27AE60" if f in ["acclimatisation_days", "hydration_level"]
              else "#8E44AD"
              for f in top["feature"]]

    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    for bar, val in zip(bars, top["mean_abs_shap"][::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Global Feature Importance\nTop 15 features by mean absolute contribution")
    legend_elements = [
        mpatches.Patch(color="#C0392B", label="Heart rate / HR trajectory"),
        mpatches.Patch(color="#2980B9", label="Temperature / humidity"),
        mpatches.Patch(color="#27AE60", label="Protective factors"),
        mpatches.Patch(color="#8E44AD", label="Other features"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    plt.tight_layout()
    out = Path(plots_dir) / "shap1_global_importance.png"
    plt.savefig(out); plt.close()
    print(f"     ✓ {out.name}")


def plot_shap_per_class(shap_stack, feature_names, plots_dir, top_n=10):
    """Per-class SHAP feature importance — shows different drivers per class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SHAP Feature Importance Per Risk Class", fontsize=13, fontweight="bold")

    for cls_idx, (label, ax) in enumerate(zip(RISK_LABELS, axes.flatten())):
        mean_abs = np.mean(np.abs(shap_stack[:, :, cls_idx]), axis=0)
        feat_imp = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
        top      = feat_imp.head(top_n)
        ax.barh(top.index[::-1], top.values[::-1],
                color=list(RISK_COLORS.values())[cls_idx], alpha=0.85, edgecolor="white")
        ax.set_title(f"{label} class — top {top_n} SHAP features")
        ax.set_xlabel("Mean |SHAP|")
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out = Path(plots_dir) / "shap2_per_class_importance.png"
    plt.savefig(out); plt.close()
    print(f"     ✓ {out.name}")


def plot_shap_dependency(shap_stack, X_test, feature_names, shap_df, plots_dir):
    """SHAP dependency plots for top 3 features (Critical class)."""
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    top3  = shap_df.head(3)["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SHAP Dependency Plots — Top 3 Features (Critical class)",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, top3):
        feat_idx  = feature_names.index(feat)
        sc = ax.scatter(X_arr[:, feat_idx], shap_stack[:, feat_idx, 3],
                        alpha=0.25, s=8, c=shap_stack[:, feat_idx, 3],
                        cmap="RdYlGn_r", rasterized=True)
        plt.colorbar(sc, ax=ax, label="SHAP (Critical)")
        ax.axhline(y=0, color="#7F8C8D", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP → Critical class")
        ax.set_title(feat)

    plt.tight_layout()
    out = Path(plots_dir) / "shap3_dependency_plots.png"
    plt.savefig(out); plt.close()
    print(f"     ✓ {out.name}")


def plot_shap_waterfall_demo(shap_stack, X_test, y_test, y_pred, feature_names, plots_dir):
    """Waterfall: why did the model predict Critical/High for these workers?"""
    X_arr         = X_test.values if hasattr(X_test, "values") else X_test
    critical_mask = (y_test == 3) & (y_pred == 3)
    high_mask     = (y_test == 2) & (y_pred == 2)
    if critical_mask.sum() == 0 or high_mask.sum() == 0:
        print("     ⚠ Waterfall skipped — insufficient correct predictions")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("SHAP Waterfall — Why did the model predict this risk level?",
                 fontsize=13, fontweight="bold")

    for ax, w_idx, label, cls_idx in [
        (axes[0], np.where(critical_mask)[0][0], "Critical", 3),
        (axes[1], np.where(high_mask)[0][0],     "High",     2),
    ]:
        shap_w  = shap_stack[w_idx, :, cls_idx]
        s_idx   = np.argsort(np.abs(shap_w))[::-1][:12]
        top_f   = [feature_names[i] for i in s_idx]
        top_s   = shap_w[s_idx]
        top_v   = X_arr[w_idx, s_idx]
        colors  = ["#C0392B" if s > 0 else "#2980B9" for s in top_s]
        ylabels = [f"{f}\n= {v:.2f}" for f, v in zip(top_f, top_v)]

        ax.barh(range(len(top_s)), top_s[::-1], color=colors[::-1],
                edgecolor="white", height=0.7)
        ax.set_yticks(range(len(top_s)))
        ax.set_yticklabels(ylabels[::-1], fontsize=9)
        ax.axvline(x=0, color="#2C3E50", linewidth=1.5)
        ax.set_xlabel(f"SHAP contribution → {label} class")
        ax.set_title(f"Worker predicted: {label}\n(red=↑ risk, blue=↓ risk)")

    plt.tight_layout()
    out = Path(plots_dir) / "shap4_waterfall_demo.png"
    plt.savefig(out); plt.close()
    print(f"     ✓ {out.name}")


def generate_shap_plots(model, X_test, y_test, y_pred, feature_names, plots_dir):
    """Generate all 4 SHAP plots. Called from run_phase3.py."""
    print("\n  Computing SHAP values (TreeExplainer)...")
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    explainer, shap_stack, shap_df = compute_shap_values(model, X_arr, feature_names)
    print("\n  Generating SHAP visualisations...")
    plot_shap_summary_bar(shap_df, plots_dir)
    plot_shap_per_class(shap_stack, feature_names, plots_dir)
    plot_shap_dependency(shap_stack, X_arr, feature_names, shap_df, plots_dir)
    plot_shap_waterfall_demo(shap_stack, X_arr, y_test, y_pred, feature_names, plots_dir)
    return shap_df
```

---

## File 3 of 4 — `visualise_phase3.py`

Five evaluation plots (confusion matrix, ROC, learning curves, probability distributions, demo scenarios). Full source in the `.py` file — see `visualise_phase3.py` in the code directory.

The five plots are:
- **plot1_confusion_matrix.png** — annotated confusion matrix with recall bars on the right. Rows = actual, columns = predicted.
- **plot2_roc_curves.png** — one-vs-rest ROC curve per class with AUC annotations and sensitivity@specificity=0.90 marker.
- **plot3_learning_curves.png** — train vs validation log-loss over boosting rounds, with early stopping marker and overfitting gap annotation.
- **plot4_class_probabilities.png** — distribution of `P(class)` separately for true positives vs other classes, showing model calibration.
- **plot5_demo_scenarios.png** — the most important demo slide: Scenario A (safe worker, no alert) vs Scenario B (high-BMI novice, extreme conditions, alert fires).

---

## File 4 of 4 — `run_phase3.py`

Entry point. Orchestrates all 9 steps in order.

```python
"""
=============================================================================
PHASE 3 RUNNER — ENTRY POINT
=============================================================================
Usage:
    python run_phase3.py
    python run_phase3.py --search optuna --trials 40
    python run_phase3.py --search default --no_shap   # quick run
    python run_phase3.py --target risk_label_future_num  # t+30 upgrade
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
import train_xgboost as tx
from visualise_phase3 import generate_eval_plots
from shap_interpreter import generate_shap_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="../data/workers_with_clusters.csv")
    parser.add_argument("--output_dir", default="../data")
    parser.add_argument("--models_dir", default="../models")
    parser.add_argument("--plots_dir",  default="../outputs/phase3_plots")
    parser.add_argument("--target",     default="risk_label_num")
    parser.add_argument("--search",     default="optuna",
                        choices=["optuna", "grid", "default"])
    parser.add_argument("--trials",     type=int, default=40)
    parser.add_argument("--no_smote",   action="store_true")
    parser.add_argument("--no_shap",    action="store_true")
    parser.add_argument("--no_plots",   action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    tx.TARGET_COL    = args.target
    tx.USE_SMOTE     = not args.no_smote
    tx.SEARCH_METHOD = args.search
    tx.OPTUNA_TRIALS = args.trials
    tx.SEED          = args.seed

    for d in [args.output_dir, args.models_dir, args.plots_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start = time.time()

    print("[1/9] Loading Phase 2 dataset...")
    X, y, feature_names = tx.load_and_prepare(args.input)

    print("\n[2/9] Stratified 70/15/15 split...")
    X_train, X_val, X_test, y_train, y_val, y_test = tx.stratified_split(X, y)

    print("\n[3/9] Class imbalance handling...")
    X_train_res, y_train_res = tx.handle_class_imbalance(X_train, y_train)

    print(f"\n[4/9] Hyperparameter tuning ({args.search})...")
    X_val_arr = X_val.values if hasattr(X_val, "values") else X_val
    if args.search == "optuna":
        best_params = tx.tune_with_optuna(X_train_res, y_train_res,
                                          X_val_arr, y_val, args.trials)
    elif args.search == "grid":
        best_params = tx.tune_with_grid(X_train_res, y_train_res, X_val_arr, y_val)
    else:
        best_params = tx.DEFAULT_PARAMS.copy()

    print(f"\n[5/9] Training final model...")
    model = tx.train_final_model(X_train_res, y_train_res, X_val_arr, y_val, best_params)

    print(f"\n[6/9] {tx.CV_FOLDS}-fold cross-validation...")
    cv_results = tx.cross_validate_model(X_train, y_train, best_params)

    print("\n[7/9] Test set evaluation...")
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    metrics, y_pred, y_prob = tx.evaluate_on_test(model, X_test_arr, y_test, feature_names)

    elapsed = time.time() - start
    print(f"\n[8/9] Saving model artefacts...")
    tx.save_model_artefacts(model, feature_names, best_params, metrics, cv_results,
                            args.models_dir)
    tx.export_phase3_report(metrics, cv_results, best_params, feature_names,
                            elapsed, args.output_dir)

    print(f"\n[9/9] Generating plots and SHAP analysis...")
    if not args.no_plots:
        generate_eval_plots(model, X_test_arr, y_test, y_pred, y_prob,
                            metrics, feature_names, args.plots_dir)
    if not args.no_shap and not args.no_plots:
        generate_shap_plots(model, X_test_arr, y_test, y_pred,
                            feature_names, args.plots_dir)

    print(f"\n{'='*65}")
    print(f"  PHASE 3 COMPLETE  |  Time: {time.time()-start:.1f}s")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}  |  "
          f"AUC: {metrics['roc_auc_macro']:.4f}  |  "
          f"Recall(High): {metrics['per_class']['High']['sensitivity_recall']:.4f}  |  "
          f"Recall(Critical): {metrics['per_class']['Critical']['sensitivity_recall']:.4f}")
    print(f"{'='*65}")
    print(f"\n  App inference sequence:")
    print(f"  1. Load heatstroke_model.pkl + feature_list.json")
    print(f"  2. Load kmeans_model.pkl + cluster_scaler.pkl (Phase 2)")
    print(f"  3. New worker onboards → assign cluster_id via kmeans_model")
    print(f"  4. Build 27-feature vector (in feature_list.json order)")
    print(f"  5. probs = model.predict_proba(features)[0]")
    print(f"  6. threshold = 0.50 × adaptive_alert_multiplier")
    print(f"  7. Alert if probs[2] >= threshold or probs[3] >= threshold")

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
```

---

## App Inference Pattern (Phase 5 Preview)

```python
import joblib, json, numpy as np

# Load all artefacts
model       = joblib.load("models/heatstroke_model.pkl")
kmeans      = joblib.load("models/kmeans_model.pkl")
scaler      = joblib.load("models/cluster_scaler.pkl")
feat_info   = json.load(open("models/feature_list.json"))
FEATURES    = feat_info["features"]    # exact 27-column order
RISK_LABELS = feat_info["risk_labels"] # ["Low","Moderate","High","Critical"]

def predict_worker_risk(worker_data: dict, adaptive_mult: float = 1.0):
    """
    worker_data : dict with keys matching FEATURES
    adaptive_mult: from Phase 2 adaptive_alert_multiplier for this worker
    """
    # 1. Assign cluster (Phase 2 integration)
    cluster_features = ["age","bmi","acclimatisation_days",
                        "metabolic_rate","hydration_level","hr_delta_t30"]
    cluster_vec = scaler.transform(
        [[worker_data[f] for f in cluster_features]]
    )
    worker_data["cluster_id"] = int(kmeans.predict(cluster_vec)[0])

    # 2. Build feature vector in exact training order
    X = np.array([[worker_data.get(f, 0.0) for f in FEATURES]])

    # 3. Get probabilities
    probs = model.predict_proba(X)[0]

    # 4. Apply adaptive threshold
    threshold = 0.50 * adaptive_mult

    # 5. Fire alert
    alert = probs[2] >= threshold or probs[3] >= threshold
    predicted_class = RISK_LABELS[int(np.argmax(probs))]

    return {
        "predicted_class": predicted_class,
        "probabilities":   dict(zip(RISK_LABELS, probs.tolist())),
        "alert_fires":     alert,
        "threshold_used":  threshold,
    }
```

---

## Viva Questions on Phase 3 — Answers

**Why did you choose Optuna over GridSearchCV?**  
GridSearchCV exhaustively evaluates every combination. For 9 hyperparameters with even 3 values each, that is 3⁹ = 19,683 model fits. Optuna uses Tree-structured Parzen Estimation (TPE), a Bayesian method that learns from previous trials and focuses sampling on promising regions of the search space. With 40 trials, Optuna typically outperforms a 200-combination grid search because it adapts.

**Why SMOTE in each cross-validation fold instead of before splitting?**  
Applying SMOTE before splitting and then cross-validating would cause data leakage. The synthetic minority samples created from the full training set contain information about the validation fold's real minority samples (because SMOTE interpolates between k-nearest neighbours). This inflates validation metrics. Applying SMOTE inside each fold — only to the training portion of that fold — keeps the validation set genuinely unseen.

**Your model gets near-perfect scores. Isn't that suspicious on synthetic data?**  
Excellent question and the right instinct. High performance on synthetic data is expected because the labels are deterministically derived from the same PHS equations that generated the features — the mapping is mathematically clean. The appropriate response is not to distrust the model, but to present the results honestly: "These metrics reflect performance on PHS-grounded synthetic data. Phase 2 of deployment would involve retraining on real field data from 50–100 workers over 2–3 months, at which point we expect some performance degradation but also generalisation to actual physiological variance." This honest framing is more impressive to a viva panel than inflated claims.

**What does `cluster_id` being the 3rd most important SHAP feature mean?**  
It means the K-Means persona from Phase 2 is genuinely predictive, not decorative. The model learned that knowing which worker type a person is — Acclimatised Veteran vs High-BMI Novice — changes the risk prediction significantly, independent of their current readings. This validates the entire Phase 2 pipeline as a meaningful contribution, not just a label assignment exercise.

**How would you deploy this model on a smartphone?**  
Export the trained XGBoost to ONNX format using `onnxmltools` (one command). Load the ONNX model in a React Native app using the ONNX Runtime mobile library. The 27-feature inference call completes in under 1ms on a mid-range Android phone. The Phase 2 K-Means and scaler are exported as ONNX pipelines the same way.
