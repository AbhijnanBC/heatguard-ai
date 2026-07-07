"""
=============================================================================
PHASE 3 — EVALUATION VISUALISATION MODULE
=============================================================================
Generates 5 evaluation plots (separate from SHAP):

  1. plot1_confusion_matrix.png   — annotated confusion matrix with recall bars
  2. plot2_roc_curves.png         — per-class ROC curves + AUC + random baseline
  3. plot3_learning_curves.png    — train vs val log-loss over boosting rounds
  4. plot4_class_probability.png  — predicted probability distributions per class
  5. plot5_demo_scenarios.png     — side-by-side demo: safe vs danger worker
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
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

RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS = {
    "Low":      "#27AE60",
    "Moderate": "#F39C12",
    "High":     "#E67E22",
    "Critical": "#C0392B",
}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, metrics: dict, plots_dir: str):
    """
    Annotated confusion matrix with:
      - Count and percentage in each cell
      - Per-class recall (sensitivity) annotations on the right
      - Colour intensity = fraction of true class predicted as that label
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        "Plot 1: Confusion Matrix\n"
        "Rows = Actual class  |  Columns = Predicted class  |  "
        "Diagonal = correct predictions",
        fontsize=13, fontweight="bold"
    )

    # Normalise by row (recall perspective)
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n_labels = len(RISK_LABELS)

    # Heatmap
    ax = axes[0]
    sns.heatmap(
        cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1,
        ax=ax, linewidths=0.5, linecolor="white",
        xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
        cbar_kws={"label": "Row-normalised fraction (recall perspective)"},
    )

    # Annotate with count + percentage
    for i in range(n_labels):
        for j in range(n_labels):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "#2C3E50"
            ax.text(j + 0.5, i + 0.5,
                    f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix (row-normalised)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Right panel: sensitivity bars
    ax2   = axes[1]
    recalls = [metrics["per_class"][l]["sensitivity_recall"] for l in RISK_LABELS]
    colors  = [RISK_COLORS[l] for l in RISK_LABELS]
    bars    = ax2.barh(RISK_LABELS, recalls, color=colors, edgecolor="white", height=0.6)
    ax2.set_xlim(0, 1.15)
    ax2.axvline(x=0.75, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.8,
                label="Min target (0.75)")
    ax2.axvline(x=0.80, color="#8E44AD", linestyle="--", linewidth=1.5, alpha=0.8,
                label="Min target Critical (0.80)")
    for bar, val in zip(bars, recalls):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax2.set_title("Sensitivity (Recall)\nper class — PRIMARY METRIC")
    ax2.set_xlabel("Recall")
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = Path(plots_dir) / "plot1_confusion_matrix.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — ROC Curves (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_test: np.ndarray, y_prob: np.ndarray,
                    metrics: dict, plots_dir: str):
    """
    ROC curves for each class (one-vs-rest).
    Shows model performance across all decision thresholds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Plot 2: ROC Curves — One-vs-Rest per Risk Class\n"
        "Shows discriminative ability across all decision thresholds. "
        "AUC = area under curve (1.0 = perfect, 0.5 = random)",
        fontsize=13, fontweight="bold"
    )

    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

    for cls_idx, (label, ax) in enumerate(zip(RISK_LABELS, axes.flatten())):
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_prob[:, cls_idx])
        roc_auc     = auc(fpr, tpr)
        color       = RISK_COLORS[label]

        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")
        ax.fill_between(fpr, tpr, alpha=0.10, color=color)

        # Mark operating point at threshold=0.5
        ax.set_title(f"{label} class — AUC = {roc_auc:.4f}")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        # Annotate sensitivity at specificity=0.90 (operational target)
        idx90 = np.searchsorted(fpr, 0.10)
        if idx90 < len(tpr):
            ax.annotate(
                f"Sens={tpr[idx90]:.2f}\n@ Spec=0.90",
                xy=(fpr[idx90], tpr[idx90]),
                xytext=(0.35, 0.25),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            )

    plt.tight_layout()
    out = Path(plots_dir) / "plot2_roc_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Learning Curves (train vs val log-loss over boosting rounds)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(model, plots_dir: str):
    """
    Training vs validation log-loss over XGBoost boosting rounds.
    Validates early stopping worked and model is not overfitting.
    """
    try:
        evals = model.evals_result()
        train_loss = evals["validation_0"]["mlogloss"]
        val_loss   = evals["validation_1"]["mlogloss"]
        rounds     = list(range(1, len(train_loss) + 1))
        best_round = model.best_iteration

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Plot 3: XGBoost Learning Curves\n"
            "Train vs validation log-loss over boosting rounds",
            fontsize=13, fontweight="bold"
        )

        # Full curve
        ax1 = axes[0]
        ax1.plot(rounds, train_loss, color="#2980B9", linewidth=1.5, label="Train log-loss")
        ax1.plot(rounds, val_loss,   color="#E74C3C", linewidth=1.5, label="Validation log-loss")
        ax1.axvline(x=best_round, color="#27AE60", linestyle="--", linewidth=1.5,
                    label=f"Best round: {best_round}")
        ax1.set_xlabel("Boosting round")
        ax1.set_ylabel("Multi-class log-loss")
        ax1.set_title("Full training run")
        ax1.legend(fontsize=9)

        # Zoomed final 30%
        ax2    = axes[1]
        zoom   = int(len(rounds) * 0.7)
        ax2.plot(rounds[zoom:], train_loss[zoom:], color="#2980B9", linewidth=1.5,
                 label="Train")
        ax2.plot(rounds[zoom:], val_loss[zoom:],   color="#E74C3C", linewidth=1.5,
                 label="Validation")
        ax2.axvline(x=best_round, color="#27AE60", linestyle="--", linewidth=1.5,
                    label=f"Best: {best_round}")
        ax2.set_xlabel("Boosting round")
        ax2.set_ylabel("Multi-class log-loss")
        ax2.set_title("Zoomed — final 30% of rounds")
        ax2.legend(fontsize=9)

        # Overfitting gap annotation
        gap = val_loss[-1] - train_loss[-1]
        ax1.text(0.98, 0.95, f"Final gap: {gap:.4f}\n(gap < 0.05 = healthy)",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        plt.tight_layout()
        out = Path(plots_dir) / "plot3_learning_curves.png"
        plt.savefig(out)
        plt.close()
        print(f"     ✓ {out.name}")

    except Exception as e:
        print(f"     ⚠ Learning curves skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Predicted probability distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_probabilities(y_test: np.ndarray, y_prob: np.ndarray, plots_dir: str):
    """
    For each risk class, plot the distribution of predicted probabilities
    separately for true-positive and false-negative predictions.
    Shows how well-calibrated and confident the model is.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Plot 4: Predicted Probability Distributions\n"
        "For each class: distribution of P(class) for true positives vs false negatives",
        fontsize=13, fontweight="bold"
    )

    for cls_idx, (label, ax) in enumerate(zip(RISK_LABELS, axes.flatten())):
        true_pos_mask = (y_test == cls_idx)
        false_neg_mask = (y_test == cls_idx)   # same mask, different prob column check

        prob_when_true = y_prob[true_pos_mask, cls_idx]   # P(class) when actually that class
        prob_all       = y_prob[:, cls_idx]

        color = RISK_COLORS[label]
        bins  = np.linspace(0, 1, 40)

        ax.hist(prob_when_true, bins=bins, alpha=0.7, color=color,
                label=f"True {label}\n(n={true_pos_mask.sum()})", density=True)
        ax.hist(prob_all[~true_pos_mask], bins=bins, alpha=0.4, color="#95A5A6",
                label=f"Other classes\n(n={(~true_pos_mask).sum()})", density=True)

        ax.axvline(x=0.5, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.8,
                   label="Threshold = 0.50")
        ax.set_xlabel(f"Predicted P({label})")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} class probability distribution")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    out = Path(plots_dir) / "plot4_class_probabilities.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Demo Scenarios (Safe worker vs Danger worker)
# ─────────────────────────────────────────────────────────────────────────────

def plot_demo_scenarios(model, feature_names: list, plots_dir: str):
    """
    The most important demo plot for the presentation.
    Shows two contrasting workers fed through the model in real time:

    Scenario A — Acclimatised Veteran, mild conditions → Low risk → no alert
    Scenario B — High-BMI Novice, extreme conditions, rising HR → Critical → alert fires

    Displays the probability bar for each risk class, the predicted class,
    and the persona-adaptive alert threshold.
    """
    import pandas as pd

    # Build scenario feature vectors matching exact training features
    # Create a base dict with all features at neutral values
    def make_base():
        return {f: 0.0 for f in feature_names}

    # Scenario A: Safe worker
    scenario_a = make_base()
    safe_vals = {
        "age": 32, "bmi": 22.5, "acclimatisation_days": 65,
        "ambient_temp": 31.0, "humidity": 45.0, "wind_speed": 2.5,
        "solar_radiation": 320.0, "metabolic_rate": 155.0, "work_hours": 2.0,
        "hydration_level": 4.0, "heart_rate": 82.0, "sweat_rate": 55.0,
        "core_temp_tre": 37.1, "heat_index": 30.5, "cluster_id": 0,
        "ambient_temp_t15": 30.2, "ambient_temp_t30": 29.5,
        "humidity_t15": 44.0, "humidity_t30": 43.5,
        "heart_rate_t15": 80.5, "heart_rate_t30": 79.0,
        "hr_delta_t15": 1.5, "hr_delta_t30": 3.0,
        "temp_delta_t15": 0.8, "temp_humidity_product": 13.95,
        "vulnerability_score": -0.55, "adaptive_alert_multiplier": 1.15,
    }
    for k, v in safe_vals.items():
        if k in scenario_a:
            scenario_a[k] = v

    # Scenario B: Danger worker
    scenario_b = make_base()
    danger_vals = {
        "age": 48, "bmi": 31.0, "acclimatisation_days": 6,
        "ambient_temp": 44.5, "humidity": 87.0, "wind_speed": 0.3,
        "solar_radiation": 920.0, "metabolic_rate": 355.0, "work_hours": 6.5,
        "hydration_level": 1.0, "heart_rate": 138.0, "sweat_rate": 118.0,
        "core_temp_tre": 39.1, "heat_index": 68.5, "cluster_id": 1,
        "ambient_temp_t15": 43.5, "ambient_temp_t30": 42.8,
        "humidity_t15": 85.0, "humidity_t30": 83.5,
        "heart_rate_t15": 130.0, "heart_rate_t30": 122.0,
        "hr_delta_t15": 8.0, "hr_delta_t30": 16.0,
        "temp_delta_t15": 1.0, "temp_humidity_product": 38.7,
        "vulnerability_score": 0.85, "adaptive_alert_multiplier": 0.80,
    }
    for k, v in danger_vals.items():
        if k in scenario_b:
            scenario_b[k] = v

    # Only use features that were in training
    df_a = pd.DataFrame([scenario_a])[feature_names]
    df_b = pd.DataFrame([scenario_b])[feature_names]

    prob_a = model.predict_proba(df_a.values)[0]
    prob_b = model.predict_proba(df_b.values)[0]
    pred_a = int(np.argmax(prob_a))
    pred_b = int(np.argmax(prob_b))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Plot 5: Live Demo Scenarios\n"
        "Scenario A (Safe Worker) vs Scenario B (Danger Worker — Alert Fires)",
        fontsize=13, fontweight="bold"
    )

    scenarios = [
        (prob_a, pred_a, "Scenario A — Acclimatised Veteran\n32yo, BMI 22.5, 65 days acclimatised\n31°C, 45% RH, HR stable at 82 bpm",
         safe_vals["adaptive_alert_multiplier"], axes[0]),
        (prob_b, pred_b, "Scenario B — High-BMI Novice\n48yo, BMI 31.0, 6 days acclimatised\n44.5°C, 87% RH, HR rising 16 bpm in 30 min",
         danger_vals["adaptive_alert_multiplier"], axes[1]),
    ]

    for probs, pred, title, mult, ax in scenarios:
        colors_bar = [RISK_COLORS[l] for l in RISK_LABELS]
        bars       = ax.barh(RISK_LABELS, probs, color=colors_bar,
                             edgecolor="white", height=0.6)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Predicted probability")
        ax.set_title(title, fontsize=10)

        # Alert threshold line (adaptive)
        threshold = 0.50 * mult
        ax.axvline(x=threshold, color="#2C3E50", linestyle="--", linewidth=2,
                   label=f"Alert threshold = {threshold:.2f}\n(×{mult} adaptive mult.)")

        for bar, prob, label in zip(bars, probs, RISK_LABELS):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.3f}", va="center", fontsize=10, fontweight="bold")

        # Bold the predicted class
        for i, (tick, label) in enumerate(zip(ax.get_yticklabels(), RISK_LABELS)):
            tick.set_fontweight("bold" if i == pred else "normal")
            tick.set_color(RISK_COLORS[label])

        # Alert box
        if probs[2] >= threshold or probs[3] >= threshold:
            ax.text(0.95, 0.05, "🔴 ALERT FIRES",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, fontweight="bold", color="#C0392B",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#FADBD8", alpha=0.9))
        else:
            ax.text(0.95, 0.05, "✅ NO ALERT",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, fontweight="bold", color="#27AE60",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#D5F5E3", alpha=0.9))

        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = Path(plots_dir) / "plot5_demo_scenarios.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_eval_plots(model, X_test, y_test, y_pred, y_prob,
                         metrics, feature_names, plots_dir):
    """Generate all 5 evaluation plots. Called from run_phase3.py."""
    print("\n  Generating evaluation visualisations...")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, metrics, plots_dir)
    plot_roc_curves(y_test, y_prob, metrics, plots_dir)
    plot_learning_curves(model, plots_dir)
    plot_class_probabilities(y_test, y_prob, plots_dir)
    plot_demo_scenarios(model, feature_names, plots_dir)

    print(f"\n  ✓ All 5 evaluation plots saved to: {plots_dir}/")
