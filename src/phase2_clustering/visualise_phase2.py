"""
=============================================================================
PHASE 2 — VISUALISATION MODULE
=============================================================================
Generates 7 diagnostic and presentation plots for Phase 2 K-Means profiling.

Plots:
  1. elbow_silhouette.png        — WCSS elbow + silhouette/DB/CH metric curves
  2. cluster_scatter_pca.png     — 2D PCA scatter coloured by cluster + risk
  3. cluster_radar.png           — Spider/radar chart of cluster feature profiles
  4. cluster_risk_heatmap.png    — Risk class distribution per cluster (heatmap)
  5. feature_distributions.png   — Violin plots of each clustering feature by cluster
  6. vulnerability_boxplot.png   — Vulnerability score distribution per cluster
  7. adaptive_threshold.png      — Adaptive alert multiplier visualisation
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import math

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

CLUSTER_FEATURES = [
    "age", "bmi", "acclimatisation_days",
    "metabolic_rate", "hydration_level", "hr_delta_t30",
]

RISK_LABELS  = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS  = {"Low": "#27AE60", "Moderate": "#F39C12",
                "High": "#E67E22", "Critical": "#C0392B"}

# Cluster colour palette (up to 6 clusters)
CLUSTER_PALETTE = ["#2980B9", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12", "#16A085"]


def _cluster_colors(k):
    return CLUSTER_PALETTE[:k]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Elbow + silhouette + Davies-Bouldin + Calinski-Harabasz
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_metrics(metrics: dict, optimal_k: int, plots_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Plot 1: K Selection Metrics — Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz\n"
        "Used to determine optimal number of worker risk profile clusters",
        fontsize=13, fontweight="bold"
    )

    k_vals = metrics["k"]
    opt_idx = k_vals.index(optimal_k)

    configs = [
        ("inertia",          "Inertia (WCSS)",           "Lower is better",  "#2980B9", False),
        ("silhouette",       "Silhouette Score",          "Higher is better", "#27AE60", True),
        ("davies_bouldin",   "Davies-Bouldin Index",      "Lower is better",  "#E74C3C", False),
        ("calinski_harabasz","Calinski-Harabasz Score",   "Higher is better", "#8E44AD", True),
    ]

    for ax, (key, title, subtitle, color, higher_better) in zip(axes.flatten(), configs):
        vals = metrics[key]
        ax.plot(k_vals, vals, marker="o", color=color, linewidth=2.2, markersize=7)
        ax.axvline(x=optimal_k, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.8)

        opt_val = vals[opt_idx]
        ax.scatter([optimal_k], [opt_val], color="#2C3E50", s=100, zorder=5)
        ax.annotate(
            f"k={optimal_k}\n({opt_val:.3f})",
            xy=(optimal_k, opt_val),
            xytext=(optimal_k + 0.4, opt_val),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
        )

        ax.set_title(f"{title}\n({subtitle})")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(title)
        ax.set_xticks(k_vals)

    plt.tight_layout()
    out = Path(plots_dir) / "plot1_elbow_metrics.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — PCA 2D Scatter (cluster + risk overlay)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_scatter_pca(df: pd.DataFrame, X_pca: np.ndarray,
                              explained: np.ndarray, k: int, plots_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Plot 2: PCA 2D Projection of Worker Clusters\n"
        "Left: coloured by cluster persona  |  Right: coloured by risk class",
        fontsize=13, fontweight="bold"
    )

    col_names = [f"PC1 ({explained[0]*100:.1f}% variance)",
                 f"PC2 ({explained[1]*100:.1f}% variance)"]

    # Left: cluster colours
    ax1 = axes[0]
    colors_cluster = _cluster_colors(k)
    unique_ids     = sorted(df["cluster_id"].unique())
    for i, cid in enumerate(unique_ids):
        mask    = df["cluster_id"].values == cid
        persona = df.loc[mask, "persona_name"].iloc[0]
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors_cluster[i], alpha=0.35, s=8, label=persona, rasterized=True,
        )
    ax1.set_xlabel(col_names[0])
    ax1.set_ylabel(col_names[1])
    ax1.set_title("Worker clusters (K-Means personas)")
    ax1.legend(fontsize=9, markerscale=3, title="Cluster persona")

    # Right: risk class colours
    ax2 = axes[1]
    for lbl in RISK_LABELS:
        mask = df["risk_label_str"].values == lbl
        if mask.sum() == 0:
            continue
        ax2.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=RISK_COLORS[lbl], alpha=0.35, s=8, label=lbl, rasterized=True,
        )
    ax2.set_xlabel(col_names[0])
    ax2.set_ylabel(col_names[1])
    ax2.set_title("Risk classes overlaid on cluster space")
    handles = [mpatches.Patch(color=RISK_COLORS[r], label=r) for r in RISK_LABELS]
    ax2.legend(handles=handles, fontsize=9, title="Risk class")

    plt.tight_layout()
    out = Path(plots_dir) / "plot2_cluster_scatter_pca.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Radar / Spider chart per cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_radar(profile_df: pd.DataFrame, k: int, plots_dir: str):
    """
    Radar chart showing the normalised centroid values for each cluster
    across all 6 clustering features. The most powerful slide for viva.
    """
    feat_labels = ["Age", "BMI", "Acclimatisation\n(days)", "Metabolic\nRate",
                   "Hydration\nLevel", "HR Δ\n(t-30)"]
    n_feat = len(CLUSTER_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Plot 3: Cluster Feature Radar Chart\n"
        "Normalised centroid values across all 6 clustering features",
        fontsize=13, fontweight="bold", y=0.98
    )

    colors_cluster = _cluster_colors(k)

    # Normalise features to [0, 1] across all workers for radar comparability
    for _, row in profile_df.iterrows():
        cid     = int(row["cluster_id"])
        persona = row["persona_name"]
        color   = colors_cluster[cid % len(colors_cluster)]

        values = [row[f"mean_{f}"] for f in CLUSTER_FEATURES]
        # Normalise using fixed physiological bounds
        bounds = [
            (18, 60),   # age
            (17, 40),   # bmi
            (0, 90),    # acclimatisation_days
            (100, 400), # metabolic_rate
            (1, 5),     # hydration_level
            (-8, 21),   # hr_delta_t30
        ]
        norm_values = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(values, bounds)]
        norm_values += norm_values[:1]

        ax.plot(angles, norm_values, color=color, linewidth=2.2, label=persona)
        ax.fill(angles, norm_values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], size=9, color="#7F8C8D")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    out = Path(plots_dir) / "plot3_cluster_radar.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Risk class distribution per cluster (heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_risk_heatmap(profile_df: pd.DataFrame, plots_dir: str):
    risk_cols = ["pct_low", "pct_moderate", "pct_high", "pct_critical"]
    display   = ["Low", "Moderate", "High", "Critical"]

    data = profile_df[["persona_name"] + risk_cols].set_index("persona_name")
    data.columns = display

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Plot 4: Risk Class Distribution within Each Cluster\n"
        "Validates that cluster assignment reflects true heat risk differences",
        fontsize=13, fontweight="bold"
    )

    # Heatmap
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="YlOrRd",
        vmin=0, vmax=100, ax=axes[0],
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "% workers in class"},
    )
    axes[0].set_title("Risk distribution heatmap (%)")
    axes[0].set_xlabel("Risk class")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", rotation=0)

    # Grouped bar chart
    ax2 = axes[1]
    n_clusters = len(data)
    n_risk     = len(display)
    x = np.arange(n_clusters)
    width = 0.18
    offsets = np.linspace(-(n_risk - 1) / 2, (n_risk - 1) / 2, n_risk) * width

    for i, (risk_lbl, offset) in enumerate(zip(display, offsets)):
        vals = data[risk_lbl].values
        bars = ax2.bar(x + offset, vals, width, color=list(RISK_COLORS.values())[i],
                       alpha=0.85, label=risk_lbl, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(data.index, fontsize=10)
    ax2.set_ylabel("% workers in risk class")
    ax2.set_title("Risk distribution bar chart")
    ax2.legend(title="Risk class", fontsize=9)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    out = Path(plots_dir) / "plot4_cluster_risk_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Violin plots of each clustering feature by cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, k: int, plots_dir: str):
    n_feat = len(CLUSTER_FEATURES)
    ncols  = 3
    nrows  = math.ceil(n_feat / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.suptitle(
        "Plot 5: Feature Distribution by Cluster (Violin + Box)\n"
        "Shows how each cluster differs across the 6 profiling features",
        fontsize=13, fontweight="bold"
    )

    feat_display = {
        "age":                 "Age (years)",
        "bmi":                 "BMI (kg/m²)",
        "acclimatisation_days":"Acclimatisation (days)",
        "metabolic_rate":      "Metabolic Rate (W)",
        "hydration_level":     "Hydration Level (1-5)",
        "hr_delta_t30":        "HR Δ from t-30min (bpm)",
    }

    unique_ids = sorted(df["cluster_id"].unique())
    palette = {str(cid): CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
               for i, cid in enumerate(unique_ids)}
    persona_map = df.drop_duplicates("cluster_id").set_index("cluster_id")["persona_name"]
    df_plot = df.copy()
    df_plot["cluster_id"] = df_plot["cluster_id"].astype(str)

    for i, feat in enumerate(CLUSTER_FEATURES):
        ax = axes.flatten()[i]
        sns.violinplot(
            data=df_plot, x="cluster_id", y=feat, palette=palette,
            ax=ax, inner="box", cut=0, linewidth=1.2,
        )
        ax.set_title(feat_display.get(feat, feat))
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(feat_display.get(feat, feat).split("(")[-1].replace(")", ""))
        ax.set_xticklabels(
            [persona_map.get(cid, f"C{cid}")[:18] for cid in unique_ids],
            rotation=20, ha="right", fontsize=9
        )

    # Hide unused subplots
    for j in range(n_feat, nrows * ncols):
        axes.flatten()[j].set_visible(False)

    plt.tight_layout()
    out = Path(plots_dir) / "plot5_feature_distributions.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Vulnerability score distribution by cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_vulnerability_boxplot(df: pd.DataFrame, k: int, plots_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Plot 6: Vulnerability Score Distribution by Cluster\n"
        "Composite score = f(BMI, Age, Acclimatisation, Metabolic Rate, Hydration, HR Δ)",
        fontsize=13, fontweight="bold"
    )

    sorted_personas = (
        df.groupby("persona_name")["vulnerability_score"].mean()
          .sort_values()
          .index.tolist()
    )
    palette_persona = {p: CLUSTER_PALETTE[i] for i, p in enumerate(sorted_personas)}

    # Box plot
    sns.boxplot(
        data=df, x="persona_name", y="vulnerability_score",
        order=sorted_personas, palette=palette_persona,
        ax=axes[0], linewidth=1.5, flierprops=dict(markersize=2, alpha=0.4),
    )
    axes[0].set_title("Vulnerability score by cluster (box)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Composite vulnerability score")
    axes[0].tick_params(axis="x", rotation=20)

    # KDE plot overlay
    ax2 = axes[1]
    for persona in sorted_personas:
        subset = df[df["persona_name"] == persona]["vulnerability_score"]
        color  = palette_persona[persona]
        subset.plot.kde(ax=ax2, color=color, linewidth=2.2, label=persona)
        ax2.axvline(subset.mean(), color=color, linestyle="--", linewidth=1, alpha=0.6)

    ax2.set_title("Vulnerability score KDE by cluster")
    ax2.set_xlabel("Composite vulnerability score")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = Path(plots_dir) / "plot6_vulnerability_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Adaptive alert threshold visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_adaptive_thresholds(df: pd.DataFrame, profile_df: pd.DataFrame, plots_dir: str):
    """
    Illustrates how alert thresholds shift per cluster in the app.
    This is the most important plot for the app demo slide.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Plot 7: Adaptive Alert Threshold by Cluster\n"
        "How the app personalises early warning based on worker risk profile",
        fontsize=13, fontweight="bold"
    )

    sorted_profiles = profile_df.sort_values("mean_vulnerability")
    personas = sorted_profiles["persona_name"].tolist()
    multipliers = []
    for persona in personas:
        mult = df[df["persona_name"] == persona]["adaptive_alert_multiplier"].mean()
        multipliers.append(round(mult, 3))

    colors = [CLUSTER_PALETTE[i] for i in range(len(personas))]

    # Bar chart of multipliers
    ax1 = axes[0]
    bars = ax1.barh(personas, multipliers, color=colors, edgecolor="white",
                    height=0.5)
    ax1.axvline(x=1.0, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Baseline threshold (1.0)")
    for bar, val in zip(bars, multipliers):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"×{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.7, 1.3)
    ax1.set_xlabel("Alert threshold multiplier")
    ax1.set_title("Adaptive multiplier per cluster\n(< 1.0 = earlier alert, > 1.0 = later alert)")
    ax1.legend(fontsize=9)

    # Effective heat index threshold per cluster (illustration)
    ax2 = axes[1]
    base_hi = 42.0   # baseline heat index alert threshold (°C)
    effective = [base_hi * m for m in multipliers]

    bars2 = ax2.barh(personas, effective, color=colors, edgecolor="white", height=0.5)
    ax2.axvline(x=base_hi, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Baseline = {base_hi}°C")
    for bar, val in zip(bars2, effective):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}°C", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Effective heat index alert threshold (°C)")
    ax2.set_title("Effective alert trigger threshold\n(lower = alert fires earlier)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = Path(plots_dir) / "plot7_adaptive_thresholds.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(df, X_pca, explained, profile_df, metrics, optimal_k, plots_dir):
    print("\n  Generating Phase 2 diagnostic visualisations...")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    k = df["cluster_id"].nunique()

    plot_elbow_metrics(metrics, optimal_k, plots_dir)
    plot_cluster_scatter_pca(df, X_pca, explained, k, plots_dir)
    plot_cluster_radar(profile_df, k, plots_dir)
    plot_cluster_risk_heatmap(profile_df, plots_dir)
    plot_feature_distributions(df, k, plots_dir)
    plot_vulnerability_boxplot(df, k, plots_dir)
    plot_adaptive_thresholds(df, profile_df, plots_dir)

    print(f"\n  ✓ All 7 plots saved to: {plots_dir}/")
