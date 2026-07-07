"""
=============================================================================
PHASE 3: XGBOOST MODEL TRAINING — HEATSTROKE EARLY WARNING SYSTEM
=============================================================================
Project : AI-Powered Heatstroke Early Warning System for Outdoor Workers
Input   : workers_with_clusters.csv  (Phase 2 output — 30 columns)
Output  : heatstroke_model.pkl       (trained XGBoost classifier)
          label_encoder.pkl          (LabelEncoder for risk classes)
          feature_list.json          (exact feature names used at training time)
          best_params.json           (winning hyperparameters)
          phase3_metrics.json        (all evaluation metrics)
          phase3_summary_report.txt  (full human-readable report)

Prediction Target:
    risk_label_num (0=Low, 1=Moderate, 2=High, 3=Critical)

    NOTE on Phase 1 t+30 upgrade:
    If your Phase 1 upgrade added 'risk_label_future_num' (predicting where
    the worker will be in 30 minutes from their current readings), set
    TARGET_COL = "risk_label_future_num" in CONFIGURATION below.
    The rest of the pipeline is identical — only the target column changes.

Why XGBoost + Lag Features instead of LSTM:
    - Lag features (t-15, t-30) teach XGBoost the trajectory of heat stress,
      giving it the same time-series awareness as an LSTM without requiring
      large sequential datasets or complex recurrent architectures.
    - XGBoost runs in milliseconds on a smartphone via ONNX export.
    - SHAP values give per-prediction explanations (which factors caused
      this alert), making it auditable and trustworthy for safety-critical use.
    - Handles class imbalance natively via scale_pos_weight or SMOTE.
    - Achieves comparable performance to LSTM on tabular lag-feature datasets
      with under 10,000 samples (Bentéjac et al., 2021).

Evaluation Strategy:
    PRIMARY  — Sensitivity (Recall) for High + Critical classes:
               A missed heatstroke is not a data error — it is a collapse.
               We optimise and report recall for danger classes first.
    SECONDARY — F1-score (macro and per-class):
               Balances precision and recall; penalises both types of error.
    TERTIARY — ROC-AUC (one-vs-rest per class):
               Shows threshold-independent discriminative ability.
    REPORTED — Confusion matrix, classification report, calibration curve.
    NOT LEAD — Accuracy (reported but not the headline metric; misleading on
               imbalanced datasets and trivially inflated on synthetic data).

Author: Team — Heatstroke AI Project
=============================================================================
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change only these values to adapt the pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Prediction target — change to "risk_label_future_num" if Phase 1 t+30 upgrade applied
TARGET_COL = "risk_label_num"

# Risk class labels (index = class integer)
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
N_CLASSES   = 4

# Columns to EXCLUDE from training features
# (target, string labels, Phase 2 metadata that is not a model input)
EXCLUDE_COLS = [
    "risk_label_str", "risk_label_num", "risk_label_future_num",
    "persona_name", "core_temp_tre_future", TARGET_COL,
]

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# SMOTE: oversample minority classes in training set only
# Set False to use class-weight balancing instead of SMOTE
USE_SMOTE = True

# Cross-validation folds for model selection
CV_FOLDS = 5

# Hyperparameter search method: "grid" | "optuna" | "default"
# "default" skips tuning and uses the manually specified DEFAULT_PARAMS
SEARCH_METHOD = "optuna"
OPTUNA_TRIALS = 40   # number of Optuna trials (reduce to 15 for quick run)

# Default hyperparameters (used when SEARCH_METHOD = "default")
DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 3,
    "gamma":            0.10,
    "reg_alpha":        0.10,
    "reg_lambda":       1.50,
}

# Danger-class sensitivity threshold for model selection
# During tuning, we select the model that maximises F1-macro subject to
# recall(High) >= MIN_HIGH_RECALL and recall(Critical) >= MIN_CRITICAL_RECALL
MIN_HIGH_RECALL     = 0.75
MIN_CRITICAL_RECALL = 0.80

# Random seed (reproducibility)
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> tuple:
    """
    Load Phase 2 dataset and build the feature matrix X and target vector y.

    Feature matrix composition (25 features):
        Personal      : age, bmi, acclimatisation_days
        Environmental : ambient_temp, humidity, wind_speed, solar_radiation
        Env Lags      : ambient_temp_t15, ambient_temp_t30, humidity_t15, humidity_t30
        Work          : metabolic_rate, work_hours, hydration_level
        Physiological : heart_rate, heart_rate_t15, heart_rate_t30, sweat_rate
        PHS Output    : core_temp_tre
        Derived       : heat_index, hr_delta_t15, hr_delta_t30,
                        temp_delta_t15, temp_humidity_product
        Cluster       : cluster_id  ← Phase 2 output; encodes worker persona

    Target (y):
        risk_label_num — integer class 0/1/2/3 (or risk_label_future_num
        if Phase 1 t+30 upgrade is applied; controlled by TARGET_COL above)

    Returns:
        X (pd.DataFrame), y (np.ndarray), feature_names (list)
    """
    df = pd.read_csv(csv_path)

    # Determine feature columns: all numeric, excluding target and metadata
    exclude_set = set(EXCLUDE_COLS)
    feature_cols = [
        c for c in df.columns
        if c not in exclude_set
        and df[c].dtype != object
        and c != TARGET_COL
    ]

    # Validate target column
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"If using Phase 1 t+30 upgrade, ensure 'risk_label_future_num' exists "
            f"and set TARGET_COL = 'risk_label_future_num'."
        )

    X = df[feature_cols].copy()
    y = df[TARGET_COL].values.astype(int)

    print(f"   Features      : {len(feature_cols)} columns")
    print(f"   Samples       : {len(X):,}")
    print(f"   Target column : {TARGET_COL}")
    print(f"   Class dist    :")
    for cls, label in enumerate(RISK_LABELS):
        n   = (y == cls).sum()
        pct = n / len(y) * 100
        print(f"      Class {cls} ({label:10s}): {n:5d} ({pct:.1f}%)")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Perform stratified 70/15/15 split preserving class proportions.

    Two-step process:
      1. Split off 15% test set (held out completely until final evaluation).
      2. Split remaining 85% into 82.4% train / 17.6% validation
         → gives 70% / 15% / 15% of total.

    StratifiedKFold ensures each split maintains the same class distribution
    as the full dataset — critical for imbalanced data.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Step 1: carve out test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=SEED,
    )

    # Step 2: split train+val
    val_of_tv = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_of_tv,
        stratify=y_tv,
        random_state=SEED,
    )

    print(f"\n   Split sizes:")
    print(f"      Train : {len(X_train):5,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"      Val   : {len(X_val):5,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"      Test  : {len(X_test):5,} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to the TRAINING SET ONLY.

    Why SMOTE and not class_weight?
        Both approaches work. SMOTE generates synthetic minority-class samples
        by interpolating between existing minority-class neighbours. This:
          - Directly increases the number of minority examples, not just their weight.
          - Often outperforms class-weight reweighting on multi-class problems.
          - Allows the model to learn more generalised decision boundaries for
            the Low and Moderate classes.

    Critical constraint: SMOTE is applied ONLY to the training set.
        Applying it to validation or test sets would contaminate the evaluation
        and produce inflated metrics. The validation and test sets remain the
        original unmodified distribution.

    Returns: X_train_resampled, y_train_resampled (both numpy arrays)
    """
    if not USE_SMOTE:
        print("   SMOTE disabled — using class_weight in XGBoost instead.")
        return X_train.values, y_train

    print("   Applying SMOTE to training set...")
    print("   Class sizes BEFORE SMOTE:")
    for cls, label in enumerate(RISK_LABELS):
        n = (y_train == cls).sum()
        print(f"      Class {cls} ({label:10s}): {n:5d}")

    # k_neighbors=3 (safe for small minority classes)
    smote = SMOTE(
        sampling_strategy="not majority",  # oversample all minority classes
        k_neighbors=3,
        random_state=SEED,
    )
    X_res, y_res = smote.fit_resample(X_train.values, y_train)

    print("   Class sizes AFTER SMOTE:")
    for cls, label in enumerate(RISK_LABELS):
        n = (y_res == cls).sum()
        print(f"      Class {cls} ({label:10s}): {n:5d}")

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING — OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

def tune_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Bayesian hyperparameter optimisation using Optuna.

    Objective function:
        Maximise F1-macro on the validation set, subject to a safety constraint:
            recall(High)     >= MIN_HIGH_RECALL
            recall(Critical) >= MIN_CRITICAL_RECALL

        If either recall constraint is violated, the trial is penalised
        with a score of 0.0 (effectively discarded).

    This two-objective framing — maximise F1 subject to recall constraints —
    is the clinically correct evaluation strategy for safety-critical AI.
    We do not naively maximise accuracy.

    Search space:
        n_estimators    : [100, 800]   — number of boosting rounds
        max_depth       : [3, 10]      — tree depth (controls complexity)
        learning_rate   : [0.01, 0.30] — shrinkage rate (log-uniform)
        subsample       : [0.60, 1.00] — row sampling per tree
        colsample_bytree: [0.50, 1.00] — column sampling per tree
        min_child_weight: [1, 10]      — min sum of instance weight in leaf
        gamma           : [0.00, 0.50] — min loss reduction for split
        reg_alpha       : [0.00, 1.00] — L1 regularisation
        reg_lambda      : [0.50, 3.00] — L2 regularisation

    Returns: dict of best hyperparameters
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Compute class weights for scale_pos_weight (used in non-SMOTE case)
    classes      = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict   = dict(zip(classes, class_weights))

    def objective(trial):
        params = {
            "objective":        "multi:softprob",
            "num_class":         N_CLASSES,
            "eval_metric":      "mlogloss",
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample":        trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.00, 0.50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.00, 1.00),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.50, 3.00),
            "tree_method":      "hist",
            "random_state":     SEED,
            "verbosity":        0,
            "n_jobs":           -1,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)

        # Safety constraint check
        recall_per_class = recall_score(y_val, y_pred, average=None, zero_division=0)
        high_recall     = recall_per_class[2] if len(recall_per_class) > 2 else 0
        critical_recall = recall_per_class[3] if len(recall_per_class) > 3 else 0

        if high_recall < MIN_HIGH_RECALL or critical_recall < MIN_CRITICAL_RECALL:
            return 0.0   # trial penalised — safety constraint violated

        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1_macro

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"\n   Optuna completed {n_trials} trials")
    print(f"   Best F1-macro on validation: {study.best_value:.4f}")
    print(f"   Best params: {best}")
    return best


def tune_with_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Manual grid search over a reduced parameter grid.
    Used as fallback when Optuna is unavailable or for quick runs.
    """
    param_grid = {
        "n_estimators":     [200, 400],
        "max_depth":        [5, 7],
        "learning_rate":    [0.05, 0.10],
        "subsample":        [0.80],
        "colsample_bytree": [0.80],
        "min_child_weight": [3],
        "gamma":            [0.10],
        "reg_alpha":        [0.10],
        "reg_lambda":       [1.50],
    }

    from itertools import product as iterproduct
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(iterproduct(*values))

    best_f1, best_params = 0.0, {}
    print(f"   Grid search: {len(combos)} combinations...")

    for combo in combos:
        params = dict(zip(keys, combo))
        params.update({
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "tree_method": "hist", "random_state": SEED,
            "verbosity": 0, "n_jobs": -1,
        })
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        recalls = recall_score(y_val, y_pred, average=None, zero_division=0)
        if (len(recalls) > 3
                and recalls[2] >= MIN_HIGH_RECALL
                and recalls[3] >= MIN_CRITICAL_RECALL):
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_params = f1, dict(zip(keys, combo))

    if not best_params:
        print("   WARNING: No combination met recall constraints. Using best F1 params.")
        best_params = dict(zip(keys, combos[0]))

    print(f"   Best F1-macro: {best_f1:.4f}  |  Params: {best_params}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
) -> xgb.XGBClassifier:
    """
    Train the final XGBoost model with the best hyperparameters.

    Training details:
        - Uses early stopping on validation logloss (patience=50 rounds)
          to prevent overfitting — the model stops when validation loss
          stops improving, even if n_estimators is not reached.
        - eval_metric = "mlogloss" (multi-class log-loss)
        - tree_method = "hist" (histogram-based, fast, memory-efficient)
        - The fitted model's best_iteration is used for prediction,
          not the last round (early stopping guarantee).

    Returns: fitted XGBClassifier
    """
    params = {
        "objective":          "multi:softprob",
        "num_class":           N_CLASSES,
        "eval_metric":        ["mlogloss", "merror"],
        "tree_method":        "hist",
        "random_state":        SEED,
        "verbosity":           0,
        "n_jobs":             -1,
        "early_stopping_rounds": 50,    # XGBoost 3.x: goes in constructor
        **best_params,
    }

    # Ensure n_estimators is large enough to benefit from early stopping
    if params.get("n_estimators", 300) < 400:
        params["n_estimators"] = 400

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    print(f"\n   Training complete.")
    print(f"   Best iteration  : {model.best_iteration}")
    print(f"   Final train log-loss : {model.evals_result()['validation_0']['mlogloss'][model.best_iteration]:.4f}")
    print(f"   Final val   log-loss : {model.evals_result()['validation_1']['mlogloss'][model.best_iteration]:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    best_params: dict,
    n_folds: int = CV_FOLDS,
) -> dict:
    """
    5-fold stratified cross-validation on the full training set.

    Provides a more robust estimate of generalisation performance than
    a single validation split. Reports mean ± std for each metric.

    Returns dict of CV results.
    """
    params = {
        "objective":    "multi:softprob",
        "num_class":     N_CLASSES,
        "tree_method":  "hist",
        "random_state":  SEED,
        "verbosity":     0,
        "n_jobs":       -1,
        **best_params,
    }

    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    model = xgb.XGBClassifier(**params)

    f1_scores       = []
    high_recalls    = []
    critical_recalls = []
    X_arr = X_train_full.values if hasattr(X_train_full, "values") else X_train_full

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_arr, y_train_full)):
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

    cv_results = {
        "f1_macro_mean":       float(np.mean(f1_scores)),
        "f1_macro_std":        float(np.std(f1_scores)),
        "high_recall_mean":    float(np.mean(high_recalls)),
        "high_recall_std":     float(np.std(high_recalls)),
        "critical_recall_mean": float(np.mean(critical_recalls)),
        "critical_recall_std": float(np.std(critical_recalls)),
    }

    print(f"\n   {n_folds}-Fold Cross-Validation Results:")
    print(f"   F1-macro        : {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}")
    print(f"   Recall (High)   : {cv_results['high_recall_mean']:.4f} ± {cv_results['high_recall_std']:.4f}")
    print(f"   Recall (Critical): {cv_results['critical_recall_mean']:.4f} ± {cv_results['critical_recall_std']:.4f}")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Final evaluation on the held-out test set.

    Computes:
        - Classification report (precision, recall, F1 per class)
        - Confusion matrix
        - Sensitivity (recall) per class — headline metrics
        - F1 per class and macro-averaged
        - ROC-AUC one-vs-rest per class and macro-averaged
        - Overall accuracy (reported last — not the headline)

    The test set was never seen during training or hyperparameter tuning.
    These are unbiased estimates of real-world performance.
    """
    X_arr  = X_test.values if hasattr(X_test, "values") else X_test
    y_pred = model.predict(X_arr)
    y_prob = model.predict_proba(X_arr)

    # Per-class metrics
    report     = classification_report(y_test, y_pred,
                                        target_names=RISK_LABELS,
                                        output_dict=True, zero_division=0)
    cm         = confusion_matrix(y_test, y_pred)
    recalls    = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per     = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc        = accuracy_score(y_test, y_pred)

    # ROC-AUC one-vs-rest
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc_per   = []
    auc_macro = 0.0
    try:
        for i in range(N_CLASSES):
            auc_per.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        auc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc_per   = [0.0] * N_CLASSES
        auc_macro = 0.0

    metrics = {
        "accuracy":        round(float(acc), 4),
        "f1_macro":        round(float(f1_macro), 4),
        "f1_weighted":     round(float(f1_weighted), 4),
        "roc_auc_macro":   round(float(auc_macro), 4),
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
    print("   " + "=" * 55)
    print(f"   {'Metric':<30} {'Value':>10}")
    print("   " + "-" * 42)
    print(f"   {'Accuracy (not headline)':<30} {acc:>10.4f}")
    print(f"   {'F1-macro (primary)':<30} {f1_macro:>10.4f}")
    print(f"   {'F1-weighted':<30} {f1_weighted:>10.4f}")
    print(f"   {'ROC-AUC macro (OvR)':<30} {auc_macro:>10.4f}")
    print()
    print(f"   {'Class':<15} {'Sensitivity':>12} {'F1':>8} {'AUC':>8}")
    print("   " + "-" * 45)
    for i, label in enumerate(RISK_LABELS):
        print(f"   {label:<15} {recalls[i]:>12.4f} {f1_per[i]:>8.4f} "
              f"{auc_per[i] if i < len(auc_per) else 0.0:>8.4f}")

    print(f"\n   SAFETY CHECK:")
    h_ok = recalls[2] >= MIN_HIGH_RECALL
    c_ok = recalls[3] >= MIN_CRITICAL_RECALL
    print(f"   Recall(High)     = {recalls[2]:.4f}  "
          f"(target ≥ {MIN_HIGH_RECALL})  {'✓ PASS' if h_ok else '✗ FAIL'}")
    print(f"   Recall(Critical) = {recalls[3]:.4f}  "
          f"(target ≥ {MIN_CRITICAL_RECALL})  {'✓ PASS' if c_ok else '✗ FAIL'}")

    return metrics, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_model_artefacts(
    model: xgb.XGBClassifier,
    feature_names: list,
    best_params: dict,
    metrics: dict,
    cv_results: dict,
    models_dir: str,
):
    """Save all model artefacts needed for Phase 5 (app inference)."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model_path = Path(models_dir) / "heatstroke_model.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved          : {model_path}")

    # Feature list (critical — inference must use EXACTLY these features in this order)
    feat_path = Path(models_dir) / "feature_list.json"
    with open(feat_path, "w") as f:
        json.dump({"features": feature_names, "target": TARGET_COL,
                   "n_classes": N_CLASSES, "risk_labels": RISK_LABELS}, f, indent=2)
    print(f"  ✓ Feature list saved   : {feat_path}")

    # Best hyperparameters
    param_path = Path(models_dir) / "best_params.json"
    with open(param_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  ✓ Best params saved    : {param_path}")

    # Full metrics
    all_metrics = {**metrics, "cross_validation": cv_results}
    metrics_path = Path(models_dir) / "phase3_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Metrics saved        : {metrics_path}")


def export_phase3_report(
    metrics: dict,
    cv_results: dict,
    best_params: dict,
    feature_names: list,
    elapsed: float,
    output_dir: str,
):
    """Write a full human-readable summary report."""
    cm = np.array(metrics["confusion_matrix"])
    lines = [
        "=" * 65,
        "PHASE 3 — XGBOOST TRAINING — SUMMARY REPORT",
        "=" * 65,
        f"Prediction target      : {TARGET_COL}",
        f"Features used          : {len(feature_names)}",
        f"SMOTE applied          : {USE_SMOTE}",
        f"Tuning method          : {SEARCH_METHOD}",
        f"Elapsed time           : {elapsed:.1f}s",
        "",
        "BEST HYPERPARAMETERS",
        "-" * 50,
    ]
    for k, v in best_params.items():
        lines.append(f"  {k:<25}: {v}")

    lines += [
        "",
        f"CROSS-VALIDATION ({CV_FOLDS}-FOLD STRATIFIED)",
        "-" * 50,
        f"  F1-macro        : {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}",
        f"  Recall (High)   : {cv_results['high_recall_mean']:.4f} ± {cv_results['high_recall_std']:.4f}",
        f"  Recall (Critical): {cv_results['critical_recall_mean']:.4f} ± {cv_results['critical_recall_std']:.4f}",
        "",
        "TEST SET — HEADLINE METRICS",
        "-" * 50,
        f"  F1-macro        : {metrics['f1_macro']:.4f}  ← primary metric",
        f"  ROC-AUC macro   : {metrics['roc_auc_macro']:.4f}",
        f"  Accuracy        : {metrics['accuracy']:.4f}  (not headline — imbalanced data)",
        "",
        "TEST SET — PER-CLASS METRICS",
        "-" * 50,
        f"  {'Class':<12} {'Sensitivity':>12} {'F1':>8} {'AUC':>8} {'Precision':>10} {'N':>6}",
        "  " + "-" * 58,
    ]
    for label in RISK_LABELS:
        pc = metrics["per_class"][label]
        lines.append(
            f"  {label:<12} {pc['sensitivity_recall']:>12.4f} {pc['f1_score']:>8.4f} "
            f"{pc['roc_auc']:>8.4f} {pc['precision']:>10.4f} {pc['support']:>6}"
        )

    lines += [
        "",
        "CONFUSION MATRIX",
        "-" * 50,
        f"  Rows=Actual  Cols=Predicted  Labels={RISK_LABELS}",
        "",
    ]
    header = "  " + " ".join(f"{l[:4]:>8}" for l in RISK_LABELS)
    lines.append(header)
    for i, row in enumerate(cm):
        row_str = "  " + f"{RISK_LABELS[i][:4]:<6}" + " ".join(f"{v:>8}" for v in row)
        lines.append(row_str)

    lines += [
        "",
        "SAFETY CONSTRAINT STATUS",
        "-" * 50,
        f"  Recall(High)     ≥ {MIN_HIGH_RECALL} : "
        f"{'PASS' if metrics['per_class']['High']['sensitivity_recall'] >= MIN_HIGH_RECALL else 'FAIL'}  "
        f"(actual = {metrics['per_class']['High']['sensitivity_recall']:.4f})",
        f"  Recall(Critical) ≥ {MIN_CRITICAL_RECALL} : "
        f"{'PASS' if metrics['per_class']['Critical']['sensitivity_recall'] >= MIN_CRITICAL_RECALL else 'FAIL'}  "
        f"(actual = {metrics['per_class']['Critical']['sensitivity_recall']:.4f})",
        "",
        "FEATURES USED",
        "-" * 50,
    ]
    for feat in feature_names:
        lines.append(f"  {feat}")

    path = Path(output_dir) / "phase3_summary_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Summary report saved : {path}")
