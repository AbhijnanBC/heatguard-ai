"""
=============================================================================
PHASE 3 RUNNER — ENTRY POINT
=============================================================================
Run this script to execute the complete Phase 3 XGBoost training pipeline.

Prerequisites:
    - Phase 1 complete: workers_synthetic_5000.csv
    - Phase 2 complete: workers_with_clusters.csv (this is the input)

Usage:
    python run_phase3.py
    python run_phase3.py --search optuna --trials 40
    python run_phase3.py --search default         # skip tuning, use preset params
    python run_phase3.py --no_smote               # use class weights instead
    python run_phase3.py --target risk_label_future_num  # Phase 1 t+30 upgrade

Outputs:
    models/
        heatstroke_model.pkl      ← trained XGBoost (used by Phase 5 app)
        feature_list.json         ← exact feature order for inference
        best_params.json          ← winning hyperparameters
        phase3_metrics.json       ← all evaluation metrics
    data/
        phase3_summary_report.txt ← full human-readable report
    outputs/phase3_plots/
        plot1_confusion_matrix.png
        plot2_roc_curves.png
        plot3_learning_curves.png
        plot4_class_probabilities.png
        plot5_demo_scenarios.png
        shap1_global_importance.png
        shap2_per_class_importance.png
        shap3_dependency_plots.png
        shap4_waterfall_demo.png

Phase 1 Upgrades Handled:
    ✓ Predictive t+30 target: pass --target risk_label_future_num to train
      the model to predict where the worker will be in 30 minutes.
    ✓ Non-linear interaction: temp_humidity_product and heat_index are in
      the feature matrix, capturing the exponential synergy term.
    ✓ Dynamic lag physiology: hr_delta_t15 and hr_delta_t30 (dynamic HR
      trajectory driven by work_hours, acclimatisation, metabolic_rate)
      are in the feature matrix. SHAP will confirm these are top-ranked.
=============================================================================
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(
        description="Phase 3: XGBoost Heatstroke Risk Prediction"
    )
    parser.add_argument("--input",      type=str,
                        default="../data/workers_with_clusters.csv",
                        help="Path to Phase 2 clustered dataset")
    parser.add_argument("--output_dir", type=str, default="../data",
                        help="Directory for reports")
    parser.add_argument("--models_dir", type=str, default="../models",
                        help="Directory to save model artefacts")
    parser.add_argument("--plots_dir",  type=str, default="../outputs/phase3_plots",
                        help="Directory for evaluation plots")
    parser.add_argument("--target",     type=str, default="risk_label_num",
                        help="Target column (use 'risk_label_future_num' for t+30 upgrade)")
    parser.add_argument("--search",     type=str, default="optuna",
                        choices=["optuna", "grid", "default"],
                        help="Hyperparameter search method")
    parser.add_argument("--trials",     type=int, default=40,
                        help="Optuna trials (only used when --search=optuna)")
    parser.add_argument("--no_smote",   action="store_true",
                        help="Disable SMOTE (use class weights instead)")
    parser.add_argument("--no_shap",    action="store_true",
                        help="Skip SHAP computation (faster for quick runs)")
    parser.add_argument("--no_plots",   action="store_true",
                        help="Skip all plot generation")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Apply CLI overrides to config
    tx.TARGET_COL    = args.target
    tx.USE_SMOTE     = not args.no_smote
    tx.SEARCH_METHOD = args.search
    tx.OPTUNA_TRIALS = args.trials
    tx.SEED          = args.seed

    for d in [args.output_dir, args.models_dir, args.plots_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start = time.time()

    print("=" * 65)
    print("  PHASE 3: XGBOOST HEATSTROKE RISK PREDICTION")
    print("=" * 65)
    print(f"  Input          : {args.input}")
    print(f"  Target         : {args.target}")
    print(f"  Tuning method  : {args.search}"
          + (f"  ({args.trials} trials)" if args.search == "optuna" else ""))
    print(f"  SMOTE          : {'Yes' if not args.no_smote else 'No (class weights)'}")
    print(f"  Seed           : {args.seed}")
    print()

    # ── STEP 1: Load data ──────────────────────────────────────────────────
    print("[1/9] Loading Phase 2 dataset and building feature matrix...")
    X, y, feature_names = tx.load_and_prepare(args.input)

    # ── STEP 2: Train/val/test split ───────────────────────────────────────
    print("\n[2/9] Stratified 70/15/15 train/val/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = tx.stratified_split(X, y)

    # ── STEP 3: SMOTE ─────────────────────────────────────────────────────
    print("\n[3/9] Handling class imbalance...")
    X_train_res, y_train_res = tx.handle_class_imbalance(X_train, y_train)
    print(f"   Training set after balancing: {len(X_train_res):,} samples")

    # ── STEP 4: Hyperparameter tuning ──────────────────────────────────────
    print(f"\n[4/9] Hyperparameter tuning ({args.search})...")
    X_val_arr = X_val.values if hasattr(X_val, "values") else X_val

    if args.search == "optuna":
        best_params = tx.tune_with_optuna(
            X_train_res, y_train_res, X_val_arr, y_val,
            n_trials=args.trials,
        )
    elif args.search == "grid":
        best_params = tx.tune_with_grid(
            X_train_res, y_train_res, X_val_arr, y_val,
        )
    else:
        best_params = tx.DEFAULT_PARAMS.copy()
        print(f"   Using default parameters: {best_params}")

    # ── STEP 5: Train final model ──────────────────────────────────────────
    print(f"\n[5/9] Training final XGBoost model with best params...")
    model = tx.train_final_model(
        X_train_res, y_train_res,
        X_val_arr, y_val,
        best_params,
    )

    # ── STEP 6: Cross-validation ───────────────────────────────────────────
    print(f"\n[6/9] {tx.CV_FOLDS}-fold stratified cross-validation...")
    # CV on full training data (before SMOTE — SMOTE applied inside each fold)
    cv_results = tx.cross_validate_model(X_train, y_train, best_params)

    # ── STEP 7: Test set evaluation ────────────────────────────────────────
    print("\n[7/9] Final evaluation on held-out test set...")
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    metrics, y_pred, y_prob = tx.evaluate_on_test(
        model, X_test_arr, y_test, feature_names
    )

    # ── STEP 8: Save model artefacts ───────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n[8/9] Saving model artefacts...")
    tx.save_model_artefacts(model, feature_names, best_params,
                            metrics, cv_results, args.models_dir)
    tx.export_phase3_report(metrics, cv_results, best_params,
                            feature_names, elapsed, args.output_dir)

    # ── STEP 9: Visualisations ─────────────────────────────────────────────
    print(f"\n[9/9] Generating evaluation plots and SHAP analysis...")
    if not args.no_plots:
        generate_eval_plots(
            model, X_test_arr, y_test, y_pred, y_prob,
            metrics, feature_names, args.plots_dir
        )

    if not args.no_shap and not args.no_plots:
        generate_shap_plots(
            model, X_test_arr, y_test, y_pred,
            feature_names, args.plots_dir
        )

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    total_elapsed = time.time() - start
    print()
    print("=" * 65)
    print("  PHASE 3 COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 65)

    print("\n  VALIDATION CHECKS:")
    from pathlib import Path as P
    assert (P(args.models_dir) / "heatstroke_model.pkl").exists(), "Model not saved"
    assert (P(args.models_dir) / "feature_list.json").exists(),    "Feature list not saved"
    assert (P(args.models_dir) / "phase3_metrics.json").exists(),  "Metrics not saved"

    f1   = metrics["f1_macro"]
    auc  = metrics["roc_auc_macro"]
    hrec = metrics["per_class"]["High"]["sensitivity_recall"]
    crec = metrics["per_class"]["Critical"]["sensitivity_recall"]

    print(f"  ✓ Model saved to          : {args.models_dir}/heatstroke_model.pkl")
    print(f"  ✓ F1-macro (test)         : {f1:.4f}")
    print(f"  ✓ ROC-AUC macro (test)    : {auc:.4f}")
    print(f"  ✓ Recall(High)            : {hrec:.4f}"
          f"  {'✓ PASS' if hrec >= tx.MIN_HIGH_RECALL else '✗ FAIL'}")
    print(f"  ✓ Recall(Critical)        : {crec:.4f}"
          f"  {'✓ PASS' if crec >= tx.MIN_CRITICAL_RECALL else '✗ FAIL'}")
    print()
    print("  NEXT STEPS:")
    print("  Phase 4 → App integration:")
    print("     Load  : heatstroke_model.pkl + feature_list.json")
    print("     Load  : kmeans_model.pkl + cluster_scaler.pkl  (from Phase 2)")
    print("     At inference:")
    print("       1. Assign new worker to cluster (kmeans_model)")
    print("       2. Append cluster_id to their feature vector")
    print("       3. Call model.predict_proba(features)")
    print("       4. Apply adaptive_alert_multiplier to threshold")
    print("       5. Fire alert if P(High) or P(Critical) > threshold")

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
