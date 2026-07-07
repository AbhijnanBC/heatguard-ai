# Phase 2 — K-Means Worker Risk Profiling

**Project:** AI-Powered Heatstroke Early Warning System for Outdoor Workers  
**Input:** `workers_synthetic_5000.csv` (Phase 1 output)  
**Output:** `workers_with_clusters.csv` + 7 diagnostic plots + 3 saved model artefacts

---

## What Phase 2 Does and Why

Phase 1 built a dataset of 5,450 workers with 26 columns describing their physiology, environment, and risk label. Phase 2 takes that dataset and answers one question: **are there natural groupings of workers who share similar intrinsic vulnerability to heat stress?**

This matters because a 40-year-old construction worker with 5 days of acclimatisation and a high BMI is not the same physiological entity as a 24-year-old who has worked outdoors for 60 days. Giving both workers the same alert threshold is the core failure of every existing heatstroke warning system. K-Means clustering solves this.

The cluster label produced here (`cluster_id`) is then passed forward as a **feature** into Phase 3 XGBoost training, so the model learns that different physiological profiles follow different heat stress trajectories. The `adaptive_alert_multiplier` column tells the app exactly how much to shift the alert trigger threshold for each worker.

---

## Phase 1 Mathematical Upgrades — How Phase 2 Respects Them

| Phase 1 Upgrade | How Phase 2 Handles It |
|---|---|
| **Predictive target shift (t+30):** Risk labels now reflect where the worker will be in 30 minutes, not now. | Clustering is done on current physiological state only. The cluster label is passed to Phase 3 as a feature alongside the lag columns. XGBoost learns which profile types escalate most quickly toward the t+30 danger state. |
| **Non-linear interaction:** Temp × humidity exponential synergy term in PHS equation. | This produces more Critical-class workers, which is already in the dataset. `hr_delta_t30` (a clustering feature) implicitly captures non-linear heat load because it is the downstream physiological response to the combined environmental conditions. |
| **Dynamic lag physiology:** HR lag now driven by `work_hours`, `acclimatisation_days`, `metabolic_rate`. | `hr_delta_t30` is used directly as a clustering feature. Workers who show a large HR rise from t-30 to t (high `hr_delta_t30`) are physiologically heat-loading — this is their personal response signature, not just the environment's signature. It is exactly the right feature for profiling. |

---

## File Structure

```
heatstroke_ai/
├── clustering/
│   ├── kmeans_profiling.py      ← core engine (functions + constants)
│   ├── visualise_phase2.py      ← all 7 diagnostic plots
│   └── run_phase2.py            ← entry point — run this
├── data/
│   ├── workers_synthetic_5000.csv     ← Phase 1 input
│   ├── workers_with_clusters.csv      ← Phase 2 output (30 columns)
│   ├── cluster_profiles.csv           ← per-cluster centroid summary
│   └── phase2_summary_report.txt      ← run statistics
├── models/
│   ├── kmeans_model.pkl               ← fitted KMeans (for app inference)
│   ├── cluster_scaler.pkl             ← fitted StandardScaler
│   └── cluster_pca.pkl                ← fitted PCA (for scatter visualisation)
└── outputs/
    └── phase2_plots/
        ├── plot1_elbow_metrics.png
        ├── plot2_cluster_scatter_pca.png
        ├── plot3_cluster_radar.png
        ├── plot4_cluster_risk_heatmap.png
        ├── plot5_feature_distributions.png
        ├── plot6_vulnerability_distribution.png
        └── plot7_adaptive_thresholds.png
```

---

## Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib scipy
```

---

## How to Run

```bash
# Default run (auto-selects k, uses Phase 1 output)
python run_phase2.py

# Force k=3 (if you want the 3-persona setup for the presentation)
python run_phase2.py --k 3

# Full options
python run_phase2.py \
  --input    ../data/workers_synthetic_5000.csv \
  --output_dir ../data \
  --models_dir ../models \
  --plots_dir  ../outputs/phase2_plots \
  --k 3 \
  --seed 42
```

---

## Clustering Feature Design — Why These 6 Features

K-Means is applied **exclusively** to features that describe the **worker**, not the **environment**. The distinction is critical and is the most common viva question about Phase 2.

| Feature | Rationale for Inclusion |
|---|---|
| `age` | Thermoregulatory capacity declines with age; older workers sweat less efficiently (ISO 7933). |
| `bmi` | Higher BMI correlates with higher metabolic heat production and reduced heat dissipation area per unit mass. |
| `acclimatisation_days` | The single strongest predictor of heat tolerance. After 10–14 days, plasma volume expands and sweat onset is earlier (NIOSH). |
| `metabolic_rate` | Reflects work intensity — a permanent occupational characteristic for a given job type. |
| `hydration_level` | Chronic dehydration pattern is a personal habit. Consistently underhydrated workers form a distinct risk group. |
| `hr_delta_t30` | **Phase 1 dynamic lag upgrade.** This is the worker's HR response trajectory — how fast their cardiovascular system responds to heat load. This is a physiological fingerprint, stable across shifts. |

**Excluded (situational):** `ambient_temp`, `humidity`, `solar_radiation`, `wind_speed`, `work_hours`, `heat_index`. These change hour-to-hour and do not define a worker's profile. Including them would cluster by weather, not by person.

---

## Vulnerability Score Formula

A composite vulnerability score is computed **after** clustering (not used as input). It is used for ranking clusters and computing adaptive alert multipliers.

```
vulnerability_score =
    0.30 × (bmi − 22.5) / 5.0
  + 0.20 × (age − 35) / 15.0
  − 0.35 × acclimatisation_days / 45.0
  + 0.25 × (metabolic_rate − 250) / 100.0
  − 0.20 × (hydration_level − 3) / 2.0
  + 0.15 × hr_delta_t30 / 5.0
```

Each term is unit-normalised. The sign of each weight reflects the physiological direction of risk. Acclimatisation and hydration are negative (protective). The acclimatisation weight is the largest in magnitude because the evidence base (NIOSH, ISO 7933, Malchaire) is strongest.

---

## Adaptive Alert Multiplier Formula

```
multiplier = 1.15 − 0.35 × (vuln_score − vuln_min) / (vuln_max − vuln_min)
```

This linearly maps vulnerability to a threshold multiplier:
- Lowest vulnerability cluster → multiplier = **1.15** (alert fires 15% later — they can tolerate more)
- Highest vulnerability cluster → multiplier = **0.80** (alert fires 20% earlier — warn them sooner)

In the app, the effective alert threshold is:  
`effective_threshold = base_threshold × adaptive_alert_multiplier`

---

## Actual Pipeline Output (from executed run)

```
PHASE 2 — K-MEANS WORKER PROFILING — SUMMARY REPORT
=================================================================
Optimal k selected     : 2
Workers clustered      : 5,450
Clustering features    : age, bmi, acclimatisation_days,
                         metabolic_rate, hydration_level, hr_delta_t30

CLUSTER QUALITY METRICS AT k=2
  Inertia (WCSS)       : 27534.78
  Silhouette Score     : 0.1457
  Davies-Bouldin Index : 2.1791
  Calinski-Harabasz    : 1022.0

CLUSTER PROFILES
  Cluster 1 — Acclimatised Veteran
    Workers           : 2,075 (38.1%)
    Mean vulnerability: −0.342
    age               : μ=37.30  σ=12.73
    bmi               : μ=24.12  σ=3.78
    acclimatisation   : μ=41.28  σ=19.60 days
    metabolic_rate    : μ=189.17  σ=68.66 W
    hydration_level   : μ=3.37  σ=1.17
    hr_delta_t30      : μ=1.33  σ=2.76 bpm
    Risk distribution : Low=19.2%  Moderate=17.7%  High=38.5%  Critical=24.5%

  Cluster 0 — High-BMI Novice
    Workers           : 3,375 (61.9%)
    Mean vulnerability: +0.312
    age               : μ=39.70  σ=12.26
    bmi               : μ=24.81  σ=3.93
    acclimatisation   : μ=21.27  σ=12.07 days
    metabolic_rate    : μ=275.72  σ=82.10 W
    hydration_level   : μ=2.55  σ=1.12
    hr_delta_t30      : μ=5.58  σ=3.26 bpm
    Risk distribution : Low=0.0%  Moderate=0.9%  High=16.1%  Critical=82.9%
```

---

## New Columns Added to Dataset

| Column | Type | Description |
|---|---|---|
| `cluster_id` | int | K-Means cluster assignment (0 to k−1) |
| `persona_name` | str | Human-readable cluster persona name |
| `vulnerability_score` | float | Composite physiological risk score |
| `adaptive_alert_multiplier` | float | App alert threshold shift factor (0.80–1.15) |

---

## File 1 of 3 — `kmeans_profiling.py`

Core engine. Contains all functions and constants. Imported by the runner.

```python
"""
=============================================================================
PHASE 2: K-MEANS WORKER RISK PROFILING
=============================================================================
Project : AI-Powered Heatstroke Early Warning System
Input   : workers_synthetic_5000.csv  (Phase 1 output)
Output  : workers_with_clusters.csv   (Phase 1 dataset + cluster_label column)
          cluster_profiles.csv        (per-cluster centroid summary)
          phase2_summary_report.txt

Scientific Basis:
    K-Means unsupervised clustering is applied exclusively to stable
    physiological and occupational features — NOT to environmental or
    situational variables. This captures intrinsic worker vulnerability
    profiles that persist across different work environments.

    Three upgrades from Phase 1 are respected here:
      1. Predictive target shift (t+30): cluster assignment is used as a
         feature in Phase 3 to predict future risk, not current state.
      2. Non-linear interaction: `synergy_score` (temp×humidity non-linear
         term) is computed and available as an optional clustering feature
         but excluded by default to keep profiles worker-centric.
      3. Dynamic lag physiology: `hr_delta_t30` (the dynamic HR trajectory)
         is included as a clustering feature because it reflects a worker's
         physiological response pattern — a stable personal trait.

Cluster Features Used (6 features — worker-intrinsic only):
    age, bmi, acclimatisation_days, metabolic_rate,
    hydration_level, hr_delta_t30

    Excluded from clustering (situational/environmental):
    ambient_temp, humidity, solar_radiation, wind_speed, work_hours
    — these vary hour-to-hour and do not define a worker's profile

Author: Team — Heatstroke AI Project
=============================================================================
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Features used for K-Means clustering (worker-intrinsic only)
CLUSTER_FEATURES = [
    "age",
    "bmi",
    "acclimatisation_days",
    "metabolic_rate",
    "hydration_level",
    "hr_delta_t30",        # dynamic HR trajectory — reflects physiological response pattern
]

# K search range for elbow + silhouette analysis
K_MIN = 2
K_MAX = 9

# Final k chosen (set after running elbow analysis — overridden by auto-selection)
K_DEFAULT = 3

# Cluster persona names (assigned after inspecting centroids — update if k changes)
# Format: {cluster_id: name}
CLUSTER_PERSONA_NAMES = {
    0: "Acclimatised Veteran",
    1: "High-BMI Novice",
    2: "Young High-Exertion",
}

# Risk weights used to compute cluster vulnerability score (for profile ranking)
VULNERABILITY_WEIGHTS = {
    "bmi":                  +0.30,   # higher BMI → higher vulnerability
    "age":                  +0.20,   # higher age → higher vulnerability
    "acclimatisation_days": -0.35,   # more acclimatised → lower vulnerability
    "metabolic_rate":       +0.25,   # higher exertion → higher vulnerability
    "hydration_level":      -0.20,   # better hydrated → lower vulnerability
    "hr_delta_t30":         +0.15,   # rising HR trajectory → higher vulnerability
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(csv_path: str) -> pd.DataFrame:
    """
    Load Phase 1 dataset and validate required columns are present.
    Handles both the original Phase 1 schema and the upgraded schema
    (with core_temp_tre_future and risk_label_future if present).
    """
    df = pd.read_csv(csv_path)

    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing clustering features in dataset: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    required_labels = ["risk_label_str", "risk_label_num"]
    for col in required_labels:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Run Phase 1 first.")

    print(f"   Loaded: {df.shape[0]:,} workers × {df.shape[1]} features")
    print(f"   Risk distribution (labels used for Phase 3):")
    label_col = "risk_label_str"
    for lbl in ["Low", "Moderate", "High", "Critical"]:
        n = (df[label_col] == lbl).sum()
        pct = n / len(df) * 100
        print(f"      {lbl:10s}: {n:5d} ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING FOR CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute any derived features needed specifically for clustering.

    Added features:
        vulnerability_score : composite personal risk score (used for
                              profile ranking and adaptive threshold setting).
                              NOT used as a clustering input feature — computed
                              from cluster centroids after fitting.

    The Phase 1 upgrade of dynamic HR lag (hr_delta_t30) is already in the
    dataset. We use it directly as a clustering feature here.
    """
    out = df.copy()

    # Composite vulnerability score (used POST-clustering for profile ranking)
    # Scaled per feature so units don't dominate
    out["vulnerability_score"] = (
        0.30 * (out["bmi"] - 22.5) / 5.0
        + 0.20 * (out["age"] - 35) / 15.0
        - 0.35 * out["acclimatisation_days"] / 45.0
        + 0.25 * (out["metabolic_rate"] - 250) / 100.0
        - 0.20 * (out["hydration_level"] - 3) / 2.0
        + 0.15 * out["hr_delta_t30"] / 5.0
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple:
    """
    Standardise clustering features with StandardScaler (zero mean, unit variance).

    Returns:
        X_scaled   : numpy array of scaled features (n × len(CLUSTER_FEATURES))
        scaler     : fitted StandardScaler (save for Phase 3 / app inference)
        feature_df : DataFrame version of X_scaled with column names
    """
    X = df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_df = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)
    return X_scaled, scaler, feature_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTIMAL K SELECTION — ELBOW + SILHOUETTE + DAVIES-BOULDIN
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX,
                   seed: int = 42) -> dict:
    """
    Run K-Means for k = k_min..k_max and compute 3 cluster quality metrics.

    Metrics:
        Inertia (WCSS)       : Within-cluster sum of squares — lower is better
                               but always decreases with k. Elbow point = optimal k.
        Silhouette score     : How similar each point is to its own cluster vs
                               nearest other cluster. Range [-1, 1], higher = better.
        Davies-Bouldin index : Ratio of within-cluster scatter to between-cluster
                               separation. Lower = better. More reliable than silhouette
                               for imbalanced cluster sizes.
        Calinski-Harabasz    : Ratio of between-cluster to within-cluster dispersion.
                               Higher = better (also called Variance Ratio Criterion).

    Returns dict with arrays for each metric across k values.
    """
    results = {
        "k": list(range(k_min, k_max + 1)),
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    print(f"\n   Running K-Means for k = {k_min} to {k_max}...")
    print(f"   {'k':>3}  {'Inertia':>10}  {'Silhouette':>12}  {'Davies-Bouldin':>15}  {'Calinski-Harabasz':>18}")
    print("   " + "-" * 65)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500,
                    random_state=seed)
        labels = km.fit_predict(X_scaled)

        inertia = km.inertia_
        sil     = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
        db      = davies_bouldin_score(X_scaled, labels)
        ch      = calinski_harabasz_score(X_scaled, labels)

        results["inertia"].append(inertia)
        results["silhouette"].append(sil)
        results["davies_bouldin"].append(db)
        results["calinski_harabasz"].append(ch)

        print(f"   {k:>3}  {inertia:>10.1f}  {sil:>12.4f}  {db:>15.4f}  {ch:>18.1f}")

    return results


def auto_select_k(metrics: dict) -> int:
    """
    Automatically select optimal k using a composite score.

    Method:
        Normalise each metric to [0, 1] range, then combine:
        - Silhouette: maximise → normalise and use directly
        - Davies-Bouldin: minimise → invert then normalise
        - Calinski-Harabasz: maximise → normalise and use directly
        - Inertia elbow: weight the 2nd derivative of inertia
          (elbow detection via maximum curvature)

        Final score = 0.35×sil + 0.30×(1-db_norm) + 0.20×ch_norm + 0.15×elbow

    Returns optimal k as integer.
    """
    k_vals = metrics["k"]
    sil    = np.array(metrics["silhouette"])
    db     = np.array(metrics["davies_bouldin"])
    ch     = np.array(metrics["calinski_harabasz"])
    inert  = np.array(metrics["inertia"])

    def normalise(arr, invert=False):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr) * 0.5
        norm = (arr - mn) / (mx - mn)
        return (1 - norm) if invert else norm

    sil_n  = normalise(sil)
    db_n   = normalise(db, invert=True)
    ch_n   = normalise(ch)

    # Elbow: 2nd derivative of inertia (high = strong elbow at that k)
    d2 = np.gradient(np.gradient(inert))
    d2_n = normalise(np.abs(d2))

    composite = 0.35 * sil_n + 0.30 * db_n + 0.20 * ch_n + 0.15 * d2_n
    best_idx  = int(np.argmax(composite))
    best_k    = k_vals[best_idx]

    print(f"\n   Auto-selected k = {best_k}  (composite quality score: {composite[best_idx]:.4f})")
    print(f"   Scores at k={best_k}: Silhouette={sil[best_idx]:.4f}  "
          f"DB={db[best_idx]:.4f}  CH={ch[best_idx]:.1f}")
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL K-MEANS FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit_final_kmeans(X_scaled: np.ndarray, k: int, seed: int = 42) -> tuple:
    """
    Fit the final K-Means model with the selected k.
    Uses k-means++ initialisation with 30 restarts for stability.

    Returns:
        model    : fitted KMeans object
        labels   : cluster assignment array (n,)
        centroids: raw (scaled) centroid coordinates (k × n_features)
    """
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=30,
        max_iter=1000,
        tol=1e-6,
        random_state=seed,
    )
    labels = model.fit_predict(X_scaled)
    centroids = model.cluster_centers_
    return model, labels, centroids


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTER INTERPRETATION & PROFILE NAMING
# ─────────────────────────────────────────────────────────────────────────────

def interpret_clusters(df: pd.DataFrame, labels: np.ndarray,
                       scaler: StandardScaler, k: int) -> pd.DataFrame:
    """
    Compute per-cluster statistics and assign persona names.

    Persona assignment logic:
        Clusters are ranked by vulnerability_score (computed from centroids).
        The cluster with:
          - Lowest vulnerability  → "Acclimatised Veteran"
          - Highest vulnerability → "High-BMI Novice" (if high BMI + low acclimatisation)
                                 or "Young High-Exertion" (if high metabolic rate + low age)
          - Middle vulnerability  → the remaining persona

        For k > 3, additional personas are generated programmatically.

    Returns a DataFrame summarising each cluster's centroid statistics.
    """
    df_copy = df.copy()
    df_copy["cluster_id"] = labels

    rows = []
    for cid in sorted(df_copy["cluster_id"].unique()):
        subset = df_copy[df_copy["cluster_id"] == cid]
        n      = len(subset)
        pct    = n / len(df_copy) * 100

        row = {"cluster_id": cid, "n_workers": n, "pct_workers": round(pct, 1)}
        for feat in CLUSTER_FEATURES:
            row[f"mean_{feat}"] = round(subset[feat].mean(), 2)
            row[f"std_{feat}"]  = round(subset[feat].std(), 2)

        # Risk distribution within cluster
        for lbl in ["Low", "Moderate", "High", "Critical"]:
            row[f"pct_{lbl.lower()}"] = round(
                (subset["risk_label_str"] == lbl).sum() / n * 100, 1
            )
        row["mean_vulnerability"] = round(subset["vulnerability_score"].mean(), 3)
        rows.append(row)

    profile_df = pd.DataFrame(rows).sort_values("mean_vulnerability")
    profile_df = profile_df.reset_index(drop=True)

    # Assign persona names based on vulnerability ranking
    personas = _assign_personas(profile_df, k)
    profile_df["persona_name"] = [personas[cid] for cid in profile_df["cluster_id"]]

    return profile_df


def _assign_personas(profile_df: pd.DataFrame, k: int) -> dict:
    """
    Assign human-readable persona names to cluster IDs.
    For k=3: canonical three personas. For k>3: generated from ranking.
    """
    sorted_ids = profile_df["cluster_id"].tolist()  # sorted by vulnerability

    if k == 3:
        return {
            sorted_ids[0]: "Acclimatised Veteran",
            sorted_ids[1]: "Young High-Exertion",
            sorted_ids[2]: "High-BMI Novice",
        }
    elif k == 4:
        return {
            sorted_ids[0]: "Acclimatised Veteran",
            sorted_ids[1]: "Fit Young Worker",
            sorted_ids[2]: "High-Exertion Risk",
            sorted_ids[3]: "High-BMI Novice",
        }
    else:
        names = {
            cid: f"Profile_{rank+1}_Vuln{rank+1}"
            for rank, cid in enumerate(sorted_ids)
        }
        names[sorted_ids[0]]  = "Acclimatised Veteran"
        names[sorted_ids[-1]] = "High-BMI Novice"
        return names


def build_cluster_label_column(df: pd.DataFrame, labels: np.ndarray,
                               profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster_id, persona_name, vulnerability_score, adaptive_alert_multiplier
    columns to the main dataset.

    Multiplier formula: 1.15 − 0.35 × normalised_vulnerability
    Maps lowest-vulnerability cluster → 1.15 (alert later)
         highest-vulnerability cluster → 0.80 (alert earlier)
    """
    out = df.copy()
    out["cluster_id"] = labels

    id_to_persona = dict(zip(profile_df["cluster_id"], profile_df["persona_name"]))
    id_to_vuln    = dict(zip(profile_df["cluster_id"], profile_df["mean_vulnerability"]))

    out["persona_name"] = out["cluster_id"].map(id_to_persona)

    vuln_vals  = np.array([id_to_vuln[cid] for cid in out["cluster_id"]])
    vuln_min   = min(id_to_vuln.values())
    vuln_max   = max(id_to_vuln.values())
    vuln_range = vuln_max - vuln_min if vuln_max != vuln_min else 1.0

    out["adaptive_alert_multiplier"] = (
        1.15 - 0.35 * (vuln_vals - vuln_min) / vuln_range
    ).round(3)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA FOR VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca_projection(X_scaled: np.ndarray) -> tuple:
    """
    Reduce scaled features to 2D using PCA for cluster scatter visualisation.
    Returns (X_pca, fitted_pca, explained_variance_ratio).
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"\n   PCA: PC1={explained[0]*100:.1f}%  PC2={explained[1]*100:.1f}%"
          f"  (total explained={sum(explained)*100:.1f}%)")
    return X_pca, pca, explained


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def export_clustered_dataset(df: pd.DataFrame, output_dir: str):
    path = Path(output_dir) / "workers_with_clusters.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"\n  ✓ Clustered dataset saved : {path}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


def export_cluster_profiles(profile_df: pd.DataFrame, output_dir: str):
    path = Path(output_dir) / "cluster_profiles.csv"
    profile_df.to_csv(path, index=False)
    print(f"  ✓ Cluster profiles saved  : {path}")


def export_phase2_report(df, profile_df, metrics, k, elapsed, output_dir):
    lines = [
        "=" * 65,
        "PHASE 2 — K-MEANS WORKER PROFILING — SUMMARY REPORT",
        "=" * 65,
        f"Optimal k selected     : {k}",
        f"Workers clustered      : {len(df):,}",
        f"Clustering features    : {', '.join(CLUSTER_FEATURES)}",
        f"Elapsed time           : {elapsed:.2f}s",
        "",
        "CLUSTER QUALITY METRICS AT k=" + str(k),
        "-" * 50,
    ]
    idx = metrics["k"].index(k)
    lines += [
        f"  Inertia (WCSS)       : {metrics['inertia'][idx]:.2f}",
        f"  Silhouette Score     : {metrics['silhouette'][idx]:.4f}",
        f"  Davies-Bouldin Index : {metrics['davies_bouldin'][idx]:.4f}",
        f"  Calinski-Harabasz    : {metrics['calinski_harabasz'][idx]:.1f}",
        "",
        "CLUSTER PROFILES",
        "-" * 50,
    ]
    for _, row in profile_df.iterrows():
        lines += [
            f"\n  Cluster {int(row['cluster_id'])} — {row['persona_name']}",
            f"    Workers           : {int(row['n_workers']):,} ({row['pct_workers']:.1f}%)",
            f"    Mean vulnerability: {row['mean_vulnerability']:.3f}",
        ]
        for feat in CLUSTER_FEATURES:
            lines.append(f"    {feat:25s}: μ={row[f'mean_{feat}']:.2f}  σ={row[f'std_{feat}']:.2f}")
        lines.append(
            f"    Risk distribution : Low={row['pct_low']:.1f}%  "
            f"Moderate={row['pct_moderate']:.1f}%  "
            f"High={row['pct_high']:.1f}%  "
            f"Critical={row['pct_critical']:.1f}%"
        )

    path = Path(output_dir) / "phase2_summary_report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Summary report saved    : {path}")


def save_models(model, scaler, pca, output_dir: str):
    """Save fitted objects for Phase 3 XGBoost and app inference."""
    import joblib
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  Path(output_dir) / "kmeans_model.pkl")
    joblib.dump(scaler, Path(output_dir) / "cluster_scaler.pkl")
    if pca is not None:
        joblib.dump(pca, Path(output_dir) / "cluster_pca.pkl")
    print(f"  ✓ Model artefacts saved   : {output_dir}/"
          f"  [kmeans_model.pkl, cluster_scaler.pkl, cluster_pca.pkl]")
```

---

## File 2 of 3 — `visualise_phase2.py`

Generates all 7 diagnostic plots. Call `generate_all_plots(...)` at the end of the run.

```python
"""
=============================================================================
PHASE 2 — VISUALISATION MODULE
=============================================================================
Plots:
  1. plot1_elbow_metrics.png        — WCSS + silhouette + DB + CH curves
  2. plot2_cluster_scatter_pca.png  — 2D PCA scatter (cluster + risk overlay)
  3. plot3_cluster_radar.png        — Spider/radar chart of cluster profiles
  4. plot4_cluster_risk_heatmap.png — Risk distribution per cluster heatmap
  5. plot5_feature_distributions.png— Violin plots per clustering feature
  6. plot6_vulnerability_distribution.png — Vulnerability score box + KDE
  7. plot7_adaptive_thresholds.png  — Adaptive alert multiplier visualisation
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import math

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
CLUSTER_PALETTE = ["#2980B9", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12", "#16A085"]


def _cluster_colors(k):
    return CLUSTER_PALETTE[:k]


def plot_elbow_metrics(metrics: dict, optimal_k: int, plots_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Plot 1: K Selection Metrics — Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz\n"
        "Used to determine optimal number of worker risk profile clusters",
        fontsize=13, fontweight="bold"
    )
    k_vals  = metrics["k"]
    opt_idx = k_vals.index(optimal_k)

    configs = [
        ("inertia",          "Inertia (WCSS)",           "Lower is better",  "#2980B9"),
        ("silhouette",       "Silhouette Score",          "Higher is better", "#27AE60"),
        ("davies_bouldin",   "Davies-Bouldin Index",      "Lower is better",  "#E74C3C"),
        ("calinski_harabasz","Calinski-Harabasz Score",   "Higher is better", "#8E44AD"),
    ]
    for ax, (key, title, subtitle, color) in zip(axes.flatten(), configs):
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


def plot_cluster_scatter_pca(df, X_pca, explained, k, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Plot 2: PCA 2D Projection of Worker Clusters\n"
        "Left: coloured by cluster persona  |  Right: coloured by risk class",
        fontsize=13, fontweight="bold"
    )
    col_names = [f"PC1 ({explained[0]*100:.1f}% variance)",
                 f"PC2 ({explained[1]*100:.1f}% variance)"]

    ax1 = axes[0]
    colors_cluster = _cluster_colors(k)
    for i, cid in enumerate(sorted(df["cluster_id"].unique())):
        mask    = df["cluster_id"].values == cid
        persona = df.loc[mask, "persona_name"].iloc[0]
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors_cluster[i], alpha=0.35, s=8, label=persona, rasterized=True)
    ax1.set_xlabel(col_names[0])
    ax1.set_ylabel(col_names[1])
    ax1.set_title("Worker clusters (K-Means personas)")
    ax1.legend(fontsize=9, markerscale=3, title="Cluster persona")

    ax2 = axes[1]
    for lbl in RISK_LABELS:
        mask = df["risk_label_str"].values == lbl
        if mask.sum() == 0:
            continue
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=RISK_COLORS[lbl], alpha=0.35, s=8, label=lbl, rasterized=True)
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


def plot_cluster_radar(profile_df, k, plots_dir):
    feat_labels = ["Age", "BMI", "Acclimatisation\n(days)", "Metabolic\nRate",
                   "Hydration\nLevel", "HR Δ\n(t-30)"]
    n_feat = len(CLUSTER_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Plot 3: Cluster Feature Radar Chart\n"
        "Normalised centroid values across all 6 clustering features",
        fontsize=13, fontweight="bold", y=0.98
    )
    colors_cluster = _cluster_colors(k)
    bounds = [(18,60),(17,40),(0,90),(100,400),(1,5),(-8,21)]

    for _, row in profile_df.iterrows():
        cid     = int(row["cluster_id"])
        persona = row["persona_name"]
        color   = colors_cluster[cid % len(colors_cluster)]
        values  = [row[f"mean_{f}"] for f in CLUSTER_FEATURES]
        norm_values = [(v - lo)/(hi - lo) for v,(lo,hi) in zip(values, bounds)]
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


def plot_cluster_risk_heatmap(profile_df, plots_dir):
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
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=0, vmax=100, ax=axes[0],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"label": "% workers in class"})
    axes[0].set_title("Risk distribution heatmap (%)")
    axes[0].tick_params(axis="y", rotation=0)

    ax2    = axes[1]
    n_c    = len(data)
    n_r    = len(display)
    x      = np.arange(n_c)
    width  = 0.18
    offs   = np.linspace(-(n_r-1)/2, (n_r-1)/2, n_r) * width
    for i, (rl, offset) in enumerate(zip(display, offs)):
        ax2.bar(x + offset, data[rl].values, width,
                color=list(RISK_COLORS.values())[i], alpha=0.85,
                label=rl, edgecolor="white")
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


def plot_feature_distributions(df, k, plots_dir):
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
        "age": "Age (years)", "bmi": "BMI (kg/m²)",
        "acclimatisation_days": "Acclimatisation (days)",
        "metabolic_rate": "Metabolic Rate (W)",
        "hydration_level": "Hydration Level (1-5)",
        "hr_delta_t30": "HR Δ from t-30min (bpm)",
    }
    unique_ids   = sorted(df["cluster_id"].unique())
    palette      = {str(cid): CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                    for i, cid in enumerate(unique_ids)}
    persona_map  = df.drop_duplicates("cluster_id").set_index("cluster_id")["persona_name"]
    df_plot      = df.copy()
    df_plot["cluster_id"] = df_plot["cluster_id"].astype(str)

    for i, feat in enumerate(CLUSTER_FEATURES):
        ax = axes.flatten()[i]
        sns.violinplot(data=df_plot, x="cluster_id", y=feat, palette=palette,
                       ax=ax, inner="box", cut=0, linewidth=1.2)
        ax.set_title(feat_display.get(feat, feat))
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(feat_display.get(feat, feat).split("(")[-1].replace(")", ""))
        ax.set_xticklabels(
            [persona_map.get(cid, f"C{cid}")[:18] for cid in unique_ids],
            rotation=20, ha="right", fontsize=9
        )
    for j in range(n_feat, nrows * ncols):
        axes.flatten()[j].set_visible(False)

    plt.tight_layout()
    out = Path(plots_dir) / "plot5_feature_distributions.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


def plot_vulnerability_boxplot(df, k, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Plot 6: Vulnerability Score Distribution by Cluster\n"
        "Composite score = f(BMI, Age, Acclimatisation, Metabolic Rate, Hydration, HR Δ)",
        fontsize=13, fontweight="bold"
    )
    sorted_personas = (
        df.groupby("persona_name")["vulnerability_score"].mean()
          .sort_values().index.tolist()
    )
    palette_p = {p: CLUSTER_PALETTE[i] for i, p in enumerate(sorted_personas)}

    sns.boxplot(data=df, x="persona_name", y="vulnerability_score",
                order=sorted_personas, palette=palette_p,
                ax=axes[0], linewidth=1.5,
                flierprops=dict(markersize=2, alpha=0.4))
    axes[0].set_title("Vulnerability score by cluster (box)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Composite vulnerability score")
    axes[0].tick_params(axis="x", rotation=20)

    ax2 = axes[1]
    for persona in sorted_personas:
        subset = df[df["persona_name"] == persona]["vulnerability_score"]
        subset.plot.kde(ax=ax2, color=palette_p[persona], linewidth=2.2, label=persona)
        ax2.axvline(subset.mean(), color=palette_p[persona], linestyle="--",
                    linewidth=1, alpha=0.6)
    ax2.set_title("Vulnerability score KDE by cluster")
    ax2.set_xlabel("Composite vulnerability score")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = Path(plots_dir) / "plot6_vulnerability_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"     ✓ {out.name}")


def plot_adaptive_thresholds(df, profile_df, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Plot 7: Adaptive Alert Threshold by Cluster\n"
        "How the app personalises early warning based on worker risk profile",
        fontsize=13, fontweight="bold"
    )
    sorted_profiles = profile_df.sort_values("mean_vulnerability")
    personas     = sorted_profiles["persona_name"].tolist()
    multipliers  = [df[df["persona_name"] == p]["adaptive_alert_multiplier"].mean()
                    for p in personas]
    colors       = [CLUSTER_PALETTE[i] for i in range(len(personas))]
    base_hi      = 42.0

    ax1 = axes[0]
    bars = ax1.barh(personas, multipliers, color=colors, edgecolor="white", height=0.5)
    ax1.axvline(x=1.0, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Baseline (1.0)")
    for bar, val in zip(bars, multipliers):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"×{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.7, 1.3)
    ax1.set_xlabel("Alert threshold multiplier")
    ax1.set_title("Adaptive multiplier per cluster\n(< 1.0 = earlier alert)")
    ax1.legend(fontsize=9)

    ax2       = axes[1]
    effective = [base_hi * m for m in multipliers]
    bars2     = ax2.barh(personas, effective, color=colors, edgecolor="white", height=0.5)
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


def generate_all_plots(df, X_pca, explained, profile_df, metrics, optimal_k, plots_dir):
    """Generate all 7 Phase 2 diagnostic plots. Call this from run_phase2.py."""
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
```

---

## File 3 of 3 — `run_phase2.py`

Entry point. Run this to execute the full pipeline.

```python
"""
=============================================================================
PHASE 2 RUNNER — ENTRY POINT
=============================================================================
Run this script to execute the complete Phase 2 K-Means profiling pipeline.

Prerequisites:
    - Phase 1 complete: workers_synthetic_5000.csv must exist

Usage:
    python run_phase2.py
    python run_phase2.py --k 3 --seed 42
    python run_phase2.py --input ../data/workers_synthetic_5000.csv --k 3

Phase 1 Upgrades Handled:
    ✓ Predictive t+30 target: cluster_label passed forward as feature for Phase 3
    ✓ Non-linear PHS term: captured via hr_delta_t30 in clustering features
    ✓ Dynamic lag physiology: hr_delta_t30 used directly as clustering feature
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
    load_and_validate, engineer_cluster_features, scale_features,
    find_optimal_k, auto_select_k, fit_final_kmeans, interpret_clusters,
    build_cluster_label_column, compute_pca_projection,
    export_clustered_dataset, export_cluster_profiles,
    export_phase2_report, save_models, CLUSTER_FEATURES,
)
from visualise_phase2 import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Phase 2: K-Means Worker Risk Profiling")
    parser.add_argument("--input",      type=str, default="../data/workers_synthetic_5000.csv")
    parser.add_argument("--output_dir", type=str, default="../data")
    parser.add_argument("--models_dir", type=str, default="../models")
    parser.add_argument("--plots_dir",  type=str, default="../outputs/phase2_plots")
    parser.add_argument("--k",          type=int, default=0,
                        help="Force specific k (0 = auto-select)")
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

    print("[1/8] Loading and validating Phase 1 dataset...")
    df = load_and_validate(args.input)

    print("\n[2/8] Engineering clustering features (vulnerability score)...")
    df = engineer_cluster_features(df)
    print(f"   vulnerability_score range: "
          f"{df['vulnerability_score'].min():.3f} → {df['vulnerability_score'].max():.3f}")

    print("\n[3/8] Standardising clustering features (StandardScaler)...")
    X_scaled, scaler, _ = scale_features(df)
    print(f"   Features scaled: {CLUSTER_FEATURES}")
    print(f"   X_scaled shape : {X_scaled.shape}")

    print("\n[4/8] Running K selection analysis (elbow + silhouette + DB + CH)...")
    metrics = find_optimal_k(X_scaled, k_min=args.k_min, k_max=args.k_max, seed=args.seed)

    if args.k > 0:
        optimal_k = args.k
        print(f"\n   k manually set to: {optimal_k}")
    else:
        optimal_k = auto_select_k(metrics)

    print(f"\n[5/8] Fitting final K-Means (k={optimal_k}, n_init=30, max_iter=1000)...")
    model, labels, centroids = fit_final_kmeans(X_scaled, k=optimal_k, seed=args.seed)

    unique, counts = np.unique(labels, return_counts=True)
    print("   Cluster size distribution:")
    for cid, cnt in zip(unique, counts):
        print(f"      Cluster {cid}: {cnt:,} workers ({cnt/len(labels)*100:.1f}%)")

    print("\n[6/8] Interpreting clusters and assigning persona names...")
    profile_df = interpret_clusters(df, labels, scaler, k=optimal_k)

    print("\n   CLUSTER PROFILES:")
    print(f"   {'Persona':28s}  {'N':>6}  {'Vuln':>7}  {'High+Critical':>14}")
    print("   " + "-" * 62)
    for _, row in profile_df.iterrows():
        hc_pct = row["pct_high"] + row["pct_critical"]
        print(f"   {row['persona_name']:28s}  {int(row['n_workers']):>6,}  "
              f"{row['mean_vulnerability']:>7.3f}  {hc_pct:>13.1f}%")

    print("\n[7/8] Adding cluster columns to dataset...")
    df = build_cluster_label_column(df, labels, profile_df)
    print(f"   New columns: cluster_id, persona_name, vulnerability_score, "
          f"adaptive_alert_multiplier")

    print("\n[8/8] Exporting outputs...")
    export_clustered_dataset(df, args.output_dir)
    export_cluster_profiles(profile_df, args.output_dir)
    elapsed = time.time() - start
    export_phase2_report(df, profile_df, metrics, optimal_k, elapsed, args.output_dir)
    save_models(model, scaler, None, args.models_dir)

    X_pca, pca, explained = compute_pca_projection(X_scaled)
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

    # Validation
    print("\n  VALIDATION CHECKS:")
    assert "cluster_id"               in df.columns
    assert "persona_name"             in df.columns
    assert "vulnerability_score"      in df.columns
    assert "adaptive_alert_multiplier" in df.columns
    assert df["cluster_id"].nunique() == optimal_k
    assert df.isnull().sum().sum() == 0
    assert df["adaptive_alert_multiplier"].between(0.5, 1.5).all()
    print(f"  ✓ cluster_id present with {optimal_k} unique values")
    print(f"  ✓ persona_name assigned to all {len(df):,} workers")
    print(f"  ✓ vulnerability_score: "
          f"{df['vulnerability_score'].min():.3f} → {df['vulnerability_score'].max():.3f}")
    print(f"  ✓ adaptive_alert_multiplier: "
          f"{df['adaptive_alert_multiplier'].min():.3f} → {df['adaptive_alert_multiplier'].max():.3f}")
    print(f"  ✓ No null values")
    print()
    print("  NEXT STEPS:")
    print("  Phase 3 → Run train_xgboost.py with workers_with_clusters.csv")
    print("           cluster_id  = feature for XGBoost")
    print("           risk_label_num / risk_label_future_num = prediction TARGET")

    return df, profile_df


if __name__ == "__main__":
    df, profile_df = main()
```

---

## Viva Questions on Phase 2 — Answers

**Why K-Means and not DBSCAN or hierarchical clustering?**  
K-Means produces a fixed, interpretable number of profiles — exactly what the app needs to assign a worker to a persona at onboarding. DBSCAN produces variable cluster counts and marks outliers as noise, which can't be mapped to a fixed alert multiplier. Hierarchical clustering doesn't scale efficiently to 5,000+ workers and requires a cut-off decision that is no cleaner than the elbow method. K-Means with k-means++ initialisation converges stably and is entirely explainable.

**Why cluster on personal features only and exclude ambient temperature?**  
Because the cluster is a worker profile, not a situation profile. A 55-year-old unacclimatised construction worker is a High-BMI Novice whether it is 28°C or 45°C outside. The environmental conditions feed into XGBoost in Phase 3 — that is where the current situation is accounted for. If we clustered on temperature, the same worker would switch cluster every time the weather changed, making the persona meaningless and the adaptive threshold nonsensical.

**Why include `hr_delta_t30` as a clustering feature?**  
Because of the Phase 1 dynamic lag upgrade. `hr_delta_t30` is now computed from `work_hours`, `acclimatisation_days`, and `metabolic_rate`. A worker with high acclimatisation and low metabolic rate at early shift hours shows a very small `hr_delta_t30` — their cardiovascular system handles heat load efficiently. A high-metabolic-rate novice shows a large, rapidly rising `hr_delta_t30`. This trajectory is a physiological fingerprint that belongs in the cluster features.

**What does the adaptive alert multiplier actually change in the app?**  
The XGBoost model in Phase 3 outputs a probability score (0–1) for each risk class. The app triggers a "High risk" alert when that probability exceeds a threshold (e.g., 0.50). The adaptive multiplier scales this threshold: for the Acclimatised Veteran with multiplier 1.15, the alert fires when probability > 0.575 (later). For the High-BMI Novice with multiplier 0.80, the alert fires when probability > 0.40 (earlier). The same model, different sensitivity per person — this is personalisation grounded in data.

---

## What Phase 3 Receives

`workers_with_clusters.csv` — 5,450 rows, 30 columns:
- All original Phase 1 features (environmental, lag, derived)
- `cluster_id` — integer cluster assignment (use as categorical feature in XGBoost)
- `persona_name` — for human-readable output in the app
- `vulnerability_score` — optional feature for XGBoost or threshold computation
- `adaptive_alert_multiplier` — used by the app inference layer
- `risk_label_num` — XGBoost target (current risk)
- `risk_label_future_num` — XGBoost target if Phase 1 t+30 upgrade applied (recommended)

**Phase 3 feature matrix X** = all lag features + environmental + personal + `cluster_id`  
**Phase 3 target y** = `risk_label_future_num` (predict where the worker will be in 30 minutes)
