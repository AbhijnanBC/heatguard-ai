"""
=============================================================================
PHASE 1 RUNNER — ENTRY POINT [UPDATED]
=============================================================================
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from generate_synthetic import (
    sample_worker_parameters, compute_phs_core_temperature, compute_future_core_temperature,
    compute_heart_rate, compute_sweat_rate, assign_risk_labels,
    inject_gaussian_noise, build_lag_features, add_derived_features,
    reorder_columns, ensure_minimum_class_representation, RISK_LABELS
)
from visualise_phase1 import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description="Phase 1: PHS Synthetic Data Generation Pipeline")
    parser.add_argument("--n_workers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--plots_dir", type=str, default="./outputs/phase1_plots")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  PHASE 1: PREDICTIVE SYNTHETIC DATA GENERATION PIPELINE")
    print("=" * 65)

    print("[1/9] Sampling worker parameters...")
    df = sample_worker_parameters(args.n_workers, rng)

    print("[2/9] Computing CURRENT (t) and FUTURE (t+30) PHS core temperatures...")
    df["core_temp_tre"] = compute_phs_core_temperature(df)
    df["core_temp_tre_future"] = compute_future_core_temperature(df, df["core_temp_tre"], rng)

    print("[3/9] Computing physiological heart rate and sweat rate...")
    df["heart_rate"] = compute_heart_rate(df, rng)
    df["sweat_rate"] = compute_sweat_rate(df)

    print("[4/9] Assigning risk labels based on FUTURE (t+30) Tre thresholds...")
    risk_str, risk_num = assign_risk_labels(df["core_temp_tre_future"].values)
    df["risk_label_str"] = risk_str
    df["risk_label_num"] = risk_num

    print("[5/9] Checking class balance; augmenting underrepresented classes...")
    df = ensure_minimum_class_representation(df, rng, min_pct=0.08)

    df_pre_noise = df.copy()

    print("[6/9] Injecting Gaussian sensor noise...")
    df = inject_gaussian_noise(df, rng)

    print("[7/9] Building dynamic time-series lag features (t-15min, t-30min)...")
    df = build_lag_features(df, rng)

    print("[8/9] Computing derived features...")
    df = add_derived_features(df)

    df = reorder_columns(df)
    csv_path = Path(args.output_dir) / "workers_synthetic_5000.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n  ✓ Predictive Dataset saved : {csv_path}")

    if not args.no_plots:
        generate_all_plots(df, df_pre_noise, args.plots_dir)

    print("  ✓ Phase 1 predictive pipeline validated.")
    return df

if __name__ == "__main__":
    df = main()