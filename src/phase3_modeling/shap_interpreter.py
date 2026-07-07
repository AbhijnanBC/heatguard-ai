"""
=============================================================================
PHASE 3 — SHAP INTERPRETABILITY MODULE
=============================================================================
Uses SHAP (SHapley Additive exPlanations) to produce per-prediction
explanations for the XGBoost model. This module is used for:

  1. The SHAP summary bar chart (presentation slide 10)
  2. The SHAP beeswarm plot (shows distribution of feature impacts)
  3. Per-class SHAP waterfall (explains one High-risk and one Critical prediction)
  4. SHAP dependency plots for the top 3 features
  5. Force plot for the live demo scenario

Why SHAP?
    XGBoost is a black box — a worker sees "RED ALERT" and asks why.
    SHAP decomposes each prediction into feature-level contributions:
      "Your risk is Critical mainly because your heart rate rose 8 bpm
       in the last 30 minutes and the ambient temperature is 44°C."
    This makes the model auditable, trustworthy, and defensible in a viva.

SHAP method used: TreeExplainer (exact, not approximate — works natively
with XGBoost and is orders of magnitude faster than KernelExplainer).
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

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

RISK_LABELS    = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS    = {"Low": "#27AE60", "Moderate": "#F39C12",
                  "High": "#E67E22", "Critical": "#C0392B"}


def compute_shap_values(model, X_test: np.ndarray, feature_names: list) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    Returns:
        explainer     : fitted TreeExplainer
        shap_values   : array of shape (n_samples, n_features, n_classes)
        shap_df_mean  : DataFrame of mean |SHAP| per feature (averaged across classes)
    """
    explainer   = shap.TreeExplainer(model)
    X_arr       = X_test.values if hasattr(X_test, "values") else X_test
    shap_vals   = explainer.shap_values(X_arr)

    # shap_vals is list of n_classes arrays, each (n_samples, n_features)
    # Stack to (n_samples, n_features, n_classes)
    if isinstance(shap_vals, list):
        shap_stack = np.stack(shap_vals, axis=-1)
    else:
        shap_stack = shap_vals

    # Mean absolute SHAP per feature across all classes and samples
    mean_abs_shap = np.mean(np.abs(shap_stack).mean(axis=-1), axis=0)
    shap_df       = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n   SHAP Top 10 Features (mean |SHAP| across all classes):")
    for _, row in shap_df.head(10).iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 200)
        print(f"      {row['feature']:28s}: {row['mean_abs_shap']:.4f}  {bar}")

    return explainer, shap_stack, shap_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP PLOT 1 — Global feature importance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary_bar(shap_df: pd.DataFrame, plots_dir: str, top_n: int = 15):
    """
    Horizontal bar chart of mean |SHAP| values (global feature importance).
    This is the primary explainability slide for the presentation.
    """
    top = shap_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#C0392B" if "heart_rate" in f or "hr_delta" in f
                else "#2980B9" if "temp" in f or "humidity" in f
                else "#27AE60" if f in ["acclimatisation_days", "hydration_level"]
                else "#8E44AD"
                for f in top["feature"]]

    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)

    for bar, val in zip(bars, top["mean_abs_shap"][::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title(
        "SHAP Global Feature Importance\n"
        "Top 15 features by mean absolute contribution to risk prediction",
    )
    legend_elements = [
        mpatches.Patch(color="#C0392B", label="Heart rate / HR trajectory"),
        mpatches.Patch(color="#2980B9", label="Temperature / humidity"),
        mpatches.Patch(color="#27AE60", label="Protective factors (acclimatisation, hydration)"),
        mpatches.Patch(color="#8E44AD", label="Other features"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = Path(plots_dir) / "shap1_global_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP PLOT 2 — Per-class feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_per_class(shap_stack: np.ndarray, feature_names: list,
                         plots_dir: str, top_n: int = 10):
    """
    For each risk class, plot the top-n features by mean |SHAP| for that class.
    Shows that different features drive different risk classes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "SHAP Feature Importance Per Risk Class\n"
        "Different features drive prediction for each class",
        fontsize=13, fontweight="bold"
    )

    class_colors = [RISK_COLORS[r] for r in RISK_LABELS]

    for cls_idx, (label, ax) in enumerate(zip(RISK_LABELS, axes.flatten())):
        # Mean |SHAP| for this class
        mean_abs = np.mean(np.abs(shap_stack[:, :, cls_idx]), axis=0)
        feat_imp = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
        top      = feat_imp.head(top_n)

        ax.barh(top.index[::-1], top.values[::-1],
                color=class_colors[cls_idx], alpha=0.85, edgecolor="white")
        ax.set_title(f"{label} class — top {top_n} SHAP features")
        ax.set_xlabel("Mean |SHAP|")
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out = Path(plots_dir) / "shap2_per_class_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP PLOT 3 — SHAP dependency plots for top 3 features
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_dependency(shap_stack: np.ndarray, X_test: np.ndarray,
                          feature_names: list, shap_df: pd.DataFrame,
                          plots_dir: str):
    """
    SHAP dependency plots for the top 3 features (Critical class).
    Shows how feature value → SHAP contribution relationship looks — are they
    linear? Non-linear? Does interaction matter?
    """
    X_arr   = X_test.values if hasattr(X_test, "values") else X_test
    top3    = shap_df.head(3)["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "SHAP Dependency Plots — Top 3 Features (Critical class)\n"
        "Each point = one worker; y-axis = SHAP contribution to Critical prediction",
        fontsize=13, fontweight="bold"
    )

    for ax, feat in zip(axes, top3):
        feat_idx  = feature_names.index(feat)
        feat_vals = X_arr[:, feat_idx]
        shap_vals = shap_stack[:, feat_idx, 3]   # class 3 = Critical

        sc = ax.scatter(feat_vals, shap_vals, alpha=0.25, s=8,
                        c=shap_vals, cmap="RdYlGn_r", rasterized=True)
        plt.colorbar(sc, ax=ax, label="SHAP (Critical)")
        ax.axhline(y=0, color="#7F8C8D", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value → Critical class")
        ax.set_title(f"{feat}")

    plt.tight_layout()
    out = Path(plots_dir) / "shap3_dependency_plots.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP PLOT 4 — Waterfall for a single High-risk and Critical prediction
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_waterfall_demo(shap_stack: np.ndarray, X_test: np.ndarray,
                              y_test: np.ndarray, y_pred: np.ndarray,
                              feature_names: list, plots_dir: str):
    """
    Waterfall plots for two selected workers:
      - One correctly predicted as Critical (the demo scenario)
      - One correctly predicted as High

    Waterfall shows step-by-step how each feature pushes the prediction
    up or down from the baseline. This is the "why did the alert fire" slide.
    """
    X_arr = X_test.values if hasattr(X_test, "values") else X_test

    # Find one true Critical, one true High — both correctly predicted
    critical_mask = (y_test == 3) & (y_pred == 3)
    high_mask     = (y_test == 2) & (y_pred == 2)

    if critical_mask.sum() == 0 or high_mask.sum() == 0:
        print("     ⚠ Skipping waterfall — insufficient correct predictions found")
        return

    critical_idx = np.where(critical_mask)[0][0]
    high_idx     = np.where(high_mask)[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "SHAP Waterfall — Why did the model predict this risk level?\n"
        "Left: Critical worker  |  Right: High-risk worker",
        fontsize=13, fontweight="bold"
    )

    for ax, worker_idx, label, class_idx in [
        (axes[0], critical_idx, "Critical", 3),
        (axes[1], high_idx,     "High",     2),
    ]:
        worker_shap = shap_stack[worker_idx, :, class_idx]
        sorted_idx  = np.argsort(np.abs(worker_shap))[::-1][:12]
        top_feats   = [feature_names[i] for i in sorted_idx]
        top_shap    = worker_shap[sorted_idx]
        top_vals    = X_arr[worker_idx, sorted_idx]

        colors  = ["#C0392B" if s > 0 else "#2980B9" for s in top_shap]
        ylabels = [f"{f}\n= {v:.2f}" for f, v in zip(top_feats, top_vals)]

        ax.barh(range(len(top_shap)), top_shap[::-1],
                color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_yticks(range(len(top_shap)))
        ax.set_yticklabels(ylabels[::-1], fontsize=9)
        ax.axvline(x=0, color="#2C3E50", linewidth=1.5)
        ax.set_xlabel(f"SHAP contribution → {label} class probability")
        ax.set_title(f"Worker predicted as {label}\n(red=increases risk, blue=reduces risk)")

    plt.tight_layout()
    out = Path(plots_dir) / "shap4_waterfall_demo.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_shap_plots(model, X_test, y_test, y_pred,
                         feature_names, plots_dir):
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
