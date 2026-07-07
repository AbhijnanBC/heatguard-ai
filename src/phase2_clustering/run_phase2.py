"""
=============================================================================
PHASE 2 RUNNER — ENTRY POINT
=============================================================================
Run this script to execute the complete Phase 2 K-Means profiling pipeline.

Prerequisites:
    - Phase 1 complete: workers_synthetic_5000.csv must exist

Usage:
    python run_phase2.py
    python run_phase2.py --input ../data/workers_synthetic_5000.csv
    python run_phase2.py --input ../data/workers_synthetic_5000.csv --k 3 --seed 42

Outputs:
    data/
        workers_with_clusters.csv       ← Phase 1 dataset + cluster_id,
                                           persona_name, vulnerability_score,
                                           adaptive_alert_multiplier
        cluster_profiles.csv            ← Per-cluster centroid summary
        phase2_summary_report.txt       ← Run statistics and cluster details
    models/
        kmeans_model.pkl                ← Fitted KMeans (for app inference)
        cluster_scaler.pkl              ← Fitted StandardScaler
        cluster_pca.pkl                 ← Fitted PCA (for visualisation)
    outputs/phase2_plots/
        plot1_elbow_metrics.png
        plot2_cluster_scatter_pca.png
        plot3_cluster_radar.png
        plot4_cluster_risk_heatmap.png
        plot5_feature_distributions.png
        plot6_vulnerability_distribution.png
        plot7_adaptive_thresholds.png

Phase 1 Mathematical Upgrades Handled:
    ✓ Predictive target shift (t+30): cluster_label is used as a FEATURE
      in Phase 3 to predict future risk. We cluster on current physiological
      state and pass the label forward — the XGBoost model learns which
      profile types are most likely to escalate over the next 30 minutes.
    ✓ Non-linear interaction: hr_delta_t30 already encodes the dynamic
      physiological response that results from the non-linear PHS equation.
    ✓ Dynamic lag physiology: hr_delta_t30 is the direct output of Phase 1's
      dynamic HR lag (which uses work_hours, acclimatisation_days,
      metabolic_rate). It is the primary temporal signature used for clustering.
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kmeans_profiling import (
    load_and_validate,
    engineer_cluster_features,
    scale_features,
    find_optimal_k,
    auto_select_k,
    fit_final_kmeans,
    interpret_clusters,
    build_cluster_label_column,
    compute_pca_projection,
    export_clustered_dataset,
    export_cluster_profiles,
    export_phase2_report,
    save_models,
    CLUSTER_FEATURES,
)
from visualise_phase2 import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Phase 2: K-Means Worker Risk Profiling")
    parser.add_argument("--input",      type=str, default="../data/workers_synthetic_5000.csv",
                        help="Path to Phase 1 CSV output")
    parser.add_argument("--output_dir", type=str, default="../data",
                        help="Directory for CSV and report outputs")
    parser.add_argument("--models_dir", type=str, default="../models",
                        help="Directory to save fitted model artefacts")
    parser.add_argument("--plots_dir",  type=str, default="../outputs/phase2_plots",
                        help="Directory for visualisation outputs")
    parser.add_argument("--k",          type=int, default=0,
                        help="Force specific k (0 = auto-select from metrics)")
    parser.add_argument("--k_min",      type=int, default=2)
    parser.add_argument("--k_max",      type=int, default=9)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_plots",   action="store_true")
    args = parser.parse_args()

    for d in [args.output_dir, args.models_dir, args.plots_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start = time.time()

    print("=" * 65)
    print("  PHASE 2: K-MEANS WORKER RISK PROFILING")
    print("=" * 65)
    print(f"  Input dataset : {args.input}")
    print(f"  Seed          : {args.seed}")
    print()

    # ── STEP 1: Load data ──────────────────────────────────────────────────
    print("[1/8] Loading and validating Phase 1 dataset...")
    df = load_and_validate(args.input)

    # ── STEP 2: Feature engineering ────────────────────────────────────────
    print("\n[2/8] Engineering clustering features (vulnerability score)...")
    df = engineer_cluster_features(df)
    print(f"   vulnerability_score range: "
          f"{df['vulnerability_score'].min():.3f} → {df['vulnerability_score'].max():.3f}")

    # ── STEP 3: Scale ──────────────────────────────────────────────────────
    print("\n[3/8] Standardising clustering features (StandardScaler)...")
    X_scaled, scaler, _ = scale_features(df)
    print(f"   Features scaled: {CLUSTER_FEATURES}")
    print(f"   X_scaled shape : {X_scaled.shape}")

    # ── STEP 4: K selection ────────────────────────────────────────────────
    print("\n[4/8] Running K selection analysis (elbow + silhouette + DB + CH)...")
    metrics = find_optimal_k(X_scaled, k_min=args.k_min, k_max=args.k_max, seed=args.seed)

    if args.k > 0:
        optimal_k = args.k
        print(f"\n   k manually set to: {optimal_k}")
    else:
        optimal_k = auto_select_k(metrics)

    # ── STEP 5: Fit final model ────────────────────────────────────────────
    print(f"\n[5/8] Fitting final K-Means (k={optimal_k}, n_init=30, max_iter=1000)...")
    model, labels, centroids = fit_final_kmeans(X_scaled, k=optimal_k, seed=args.seed)

    # Distribution check
    unique, counts = np.unique(labels, return_counts=True)
    print("   Cluster size distribution:")
    for cid, cnt in zip(unique, counts):
        pct = cnt / len(labels) * 100
        print(f"      Cluster {cid}: {cnt:,} workers ({pct:.1f}%)")

    # ── STEP 6: Interpret clusters ─────────────────────────────────────────
    print("\n[6/8] Interpreting clusters and assigning persona names...")
    profile_df = interpret_clusters(df, labels, scaler, k=optimal_k)

    print("\n   CLUSTER PROFILES:")
    print(f"   {'Persona':28s}  {'N':>6}  {'Vuln':>7}  {'High+Critical':>14}")
    print("   " + "-" * 62)
    for _, row in profile_df.iterrows():
        hc_pct = row["pct_high"] + row["pct_critical"]
        print(f"   {row['persona_name']:28s}  {int(row['n_workers']):>6,}  "
              f"{row['mean_vulnerability']:>7.3f}  {hc_pct:>13.1f}%")

    # ── STEP 7: Build final dataset ────────────────────────────────────────
    print("\n[7/8] Adding cluster columns to dataset...")
    df = build_cluster_label_column(df, labels, profile_df)

    new_cols = ["cluster_id", "persona_name", "vulnerability_score",
                "adaptive_alert_multiplier"]
    print(f"   New columns added: {new_cols}")

    # ── STEP 8: Export + visualise ─────────────────────────────────────────
    print("\n[8/8] Exporting outputs...")

    export_clustered_dataset(df, args.output_dir)
    export_cluster_profiles(profile_df, args.output_dir)

    elapsed = time.time() - start
    export_phase2_report(df, profile_df, metrics, optimal_k, elapsed, args.output_dir)
    save_models(model, scaler, None, args.models_dir)  # PCA added below

    # PCA for visualisation
    X_pca, pca, explained = compute_pca_projection(X_scaled)

    # Save PCA
    joblib.dump(pca, Path(args.models_dir) / "cluster_pca.pkl")
    print(f"  ✓ PCA model saved: {args.models_dir}/cluster_pca.pkl")

    if not args.no_plots:
        generate_all_plots(df, X_pca, explained, profile_df, metrics,
                           optimal_k, args.plots_dir)

    print()
    print("=" * 65)
    print("  PHASE 2 COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 65)
    print()
    print("  VALIDATION CHECKS:")
    assert "cluster_id"               in df.columns, "cluster_id missing"
    assert "persona_name"             in df.columns, "persona_name missing"
    assert "vulnerability_score"      in df.columns, "vulnerability_score missing"
    assert "adaptive_alert_multiplier" in df.columns, "adaptive_alert_multiplier missing"
    assert df["cluster_id"].nunique() == optimal_k, "Cluster count mismatch"
    assert df.isnull().sum().sum() == 0, "Null values found in dataset"
    assert df["adaptive_alert_multiplier"].between(0.5, 1.5).all(), \
        "Adaptive multiplier out of valid range"

    print(f"  ✓ cluster_id present with {optimal_k} unique values")
    print(f"  ✓ persona_name assigned to all {len(df):,} workers")
    print(f"  ✓ vulnerability_score range: "
          f"{df['vulnerability_score'].min():.3f} → {df['vulnerability_score'].max():.3f}")
    print(f"  ✓ adaptive_alert_multiplier range: "
          f"{df['adaptive_alert_multiplier'].min():.3f} → {df['adaptive_alert_multiplier'].max():.3f}")
    print(f"  ✓ No null values in dataset")
    print()
    print("  NEXT STEPS:")
    print("  Phase 3 → Run train_xgboost.py with workers_with_clusters.csv")
    print("           The 'cluster_id' column is now a feature for XGBoost.")
    print("           The 'risk_label_num' column is the prediction TARGET.")
    print("           If Phase 1 upgrade is applied, use 'risk_label_future_num'")
    print("           as the target to predict t+30 min risk.")

    return df, profile_df


if __name__ == "__main__":
    df, profile_df = main()
