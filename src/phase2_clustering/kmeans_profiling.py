"""
=============================================================================
PHASE 2: K-MEANS WORKER RISK PROFILING
=============================================================================
Project : AI-Powered Heatstroke Early Warning System
Input   : workers_synthetic_5000.csv  (Phase 1 output)
Output  : workers_with_clusters.csv   (Phase 1 dataset + cluster_label column)
          cluster_profiles.csv        (per-cluster centroid summary)
          phase2_summary_report.txt
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

CLUSTER_FEATURES = [
    "age",
    "bmi",
    "acclimatisation_days",
    "metabolic_rate",
    "hydration_level",
    "hr_delta_t30", 
]

K_MIN = 2
K_MAX = 9

# [FIX]: Forced to 4 as per project requirements
K_DEFAULT = 4

VULNERABILITY_WEIGHTS = {
    "bmi":                  +0.30, 
    "age":                  +0.20, 
    "acclimatisation_days": -0.35, 
    "metabolic_rate":       +0.25, 
    "hydration_level":      -0.20, 
    "hr_delta_t30":         +0.15, 
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING FOR CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    X = df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_df = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)
    return X_scaled, scaler, feature_df

# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTIMAL K SELECTION (OVERRIDDEN)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX, seed: int = 42) -> dict:
    results = {
        "k": list(range(k_min, k_max + 1)),
        "inertia": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": [],
    }

    print(f"\n   Running K-Means for k = {k_min} to {k_max}...")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=seed)
        labels = km.fit_predict(X_scaled)
        
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled))))
        results["davies_bouldin"].append(davies_bouldin_score(X_scaled, labels))
        results["calinski_harabasz"].append(calinski_harabasz_score(X_scaled, labels))

    return results

def auto_select_k(metrics: dict) -> int:
    """
    [FIX]: Bypassed the auto-selection math to strictly return 4 clusters,
    ensuring the 4 planned project personas are generated.
    """
    print(f"\n   [OVERRIDE] Auto-selection bypassed. Forcing k = {K_DEFAULT} for project spec.")
    return K_DEFAULT

# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL K-MEANS FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit_final_kmeans(X_scaled: np.ndarray, k: int, seed: int = 42) -> tuple:
    model = KMeans(n_clusters=k, init="k-means++", n_init=30, max_iter=1000, tol=1e-6, random_state=seed)
    labels = model.fit_predict(X_scaled)
    return model, labels, model.cluster_centers_

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTER INTERPRETATION & PROFILE NAMING
# ─────────────────────────────────────────────────────────────────────────────

def interpret_clusters(df: pd.DataFrame, labels: np.ndarray, scaler: StandardScaler, k: int) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["cluster_id"] = labels

    rows = []
    for cid in sorted(df_copy["cluster_id"].unique()):
        subset = df_copy[df_copy["cluster_id"] == cid]
        n      = len(subset)
        row = {"cluster_id": cid, "n_workers": n, "pct_workers": round(n / len(df_copy) * 100, 1)}
        
        for feat in CLUSTER_FEATURES:
            row[f"mean_{feat}"] = round(subset[feat].mean(), 2)
            row[f"std_{feat}"]  = round(subset[feat].std(), 2)

        for lbl in ["Low", "Moderate", "High", "Critical"]:
            row[f"pct_{lbl.lower()}"] = round((subset["risk_label_str"] == lbl).sum() / n * 100, 1)
        
        row["mean_vulnerability"] = round(subset["vulnerability_score"].mean(), 3)
        rows.append(row)

    profile_df = pd.DataFrame(rows).sort_values("mean_vulnerability").reset_index(drop=True)
    personas = _assign_personas(profile_df, k)
    profile_df["persona_name"] = [personas[cid] for cid in profile_df["cluster_id"]]
    return profile_df

def _assign_personas(profile_df: pd.DataFrame, k: int) -> dict:
    sorted_ids = profile_df["cluster_id"].tolist() 
    
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
        names = {cid: f"Profile_{rank+1}_Vuln{rank+1}" for rank, cid in enumerate(sorted_ids)}
        names[sorted_ids[0]] = "Acclimatised Veteran"
        names[sorted_ids[-1]] = "High-BMI Novice"
        return names

def build_cluster_label_column(df: pd.DataFrame, labels: np.ndarray, profile_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cluster_id"] = labels

    id_to_persona = dict(zip(profile_df["cluster_id"], profile_df["persona_name"]))
    id_to_vuln    = dict(zip(profile_df["cluster_id"], profile_df["mean_vulnerability"]))

    out["persona_name"] = out["cluster_id"].map(id_to_persona)

    vuln_vals = np.array([id_to_vuln[cid] for cid in out["cluster_id"]])
    vuln_min  = min(id_to_vuln.values())
    vuln_max  = max(id_to_vuln.values())
    vuln_range = vuln_max - vuln_min if vuln_max != vuln_min else 1.0

    out["adaptive_alert_multiplier"] = (
        1.15 - 0.35 * (vuln_vals - vuln_min) / vuln_range
    ).round(3)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA FOR VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca_projection(X_scaled: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, pca.explained_variance_ratio_

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def export_clustered_dataset(df: pd.DataFrame, output_dir: str):
    path = Path(output_dir) / "workers_with_clusters.csv"
    df.to_csv(path, index=False, float_format="%.4f")

def export_cluster_profiles(profile_df: pd.DataFrame, output_dir: str):
    path = Path(output_dir) / "cluster_profiles.csv"
    profile_df.to_csv(path, index=False)

def export_phase2_report(df: pd.DataFrame, profile_df: pd.DataFrame, metrics: dict, k: int, elapsed: float, output_dir: str):
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
    
    path = Path(output_dir) / "phase2_summary_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_models(model: KMeans, scaler: StandardScaler, pca, output_dir: str):
    import joblib
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  Path(output_dir) / "kmeans_model.pkl")
    joblib.dump(scaler, Path(output_dir) / "cluster_scaler.pkl")
    joblib.dump(pca,    Path(output_dir) / "cluster_pca.pkl")