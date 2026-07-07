"""
=============================================================================
PHASE 1: PHS EQUATION-DRIVEN SYNTHETIC DATA GENERATION PIPELINE
=============================================================================
Project : AI-Powered Heatstroke Early Warning System for Outdoor Workers
Primary Research Contribution:
    A reusable, open-source synthetic data pipeline generating time-series 
    physiological data. Features non-linear PHS equations, dynamic lag 
    generation, and forward-looking (t+30) predictive risk labeling.
=============================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CORE_TEMP_BASAL       = 36.8     
MAX_CORE_TEMP         = 39.5     

SWEAT_METABOLIC_COEF  = 0.30     
SWEAT_TEMP_COEF       = 0.18     
SWEAT_HUMIDITY_COEF   = 0.005    

CORE_METABOLIC_COEF   = 0.0018  
CORE_TEMP_AMB_COEF    = 0.028   
CORE_HUMIDITY_COEF    = 0.007   
CORE_WIND_COEF        = 0.10    
CORE_RADIATION_COEF   = 0.0004  
CORE_ACCL_COEF        = 0.018   
CORE_BMI_COEF         = 0.022   
CORE_WORKHOUR_COEF    = 0.055   
CORE_AGE_COEF         = 0.003   
CORE_HYDRATION_COEF   = 0.045   

HR_BASAL              = 72      
HR_METABOLIC_COEF     = 0.15   
HR_TEMP_COEF          = 0.40   
HR_HUMIDITY_COEF      = 0.06   
HR_BMI_COEF           = 0.25   
HR_AGE_COEF           = 0.15   

NOISE_SIGMA = {
    "ambient_temp":  0.50,   
    "humidity":      2.00,   
    "heart_rate":    2.50,   
    "wind_speed":    0.20,   
}

RISK_THRESHOLDS = {"Low": (0, 37.5), "Moderate": (37.5, 38.0), "High": (38.0, 38.5), "Critical": (38.5, 99.0)}
RISK_LABELS     = ["Low", "Moderate", "High", "Critical"]
RISK_LABEL_MAP  = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
RISK_COLORS = {"Low": "#27AE60", "Moderate": "#F39C12", "High": "#E67E22", "Critical": "#C0392B"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETER SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
def sample_worker_parameters(n: int, rng: np.random.Generator) -> pd.DataFrame:
    params = {}
    params["age"]              = rng.integers(18, 61, size=n).astype(float)
    params["bmi"]              = np.clip(rng.normal(24.5, 4.0, size=n), 17.0, 40.0)
    params["acclimatisation_days"] = (rng.beta(2, 5, size=n) * 90).round()
    params["ambient_temp"]     = rng.uniform(28.0, 48.0, size=n)
    params["humidity"]         = (rng.beta(3, 2, size=n) * 70 + 25).clip(25, 95)
    params["wind_speed"]       = np.clip(rng.exponential(1.5, size=n), 0, 8)
    params["solar_radiation"]  = rng.uniform(50, 1000, size=n)
    params["metabolic_rate"]   = rng.uniform(100, 400, size=n)   
    params["work_hours"]       = rng.uniform(0, 8, size=n)
    hydration_probs            = [0.15, 0.30, 0.30, 0.15, 0.10]
    params["hydration_level"]  = rng.choice([1, 2, 3, 4, 5], size=n, p=hydration_probs).astype(float)
    return pd.DataFrame(params)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHS MODEL EQUATIONS [FIX 2: NON-LINEARITY ADDED]
# ─────────────────────────────────────────────────────────────────────────────
def compute_phs_core_temperature(df: pd.DataFrame) -> np.ndarray:
    """Calculates CURRENT core temperature with non-linear interaction terms."""
    # [FIX 2] Non-linear penalty: High Heat + High Humidity = Exponential Strain
    interaction_penalty = np.where(
        (df["ambient_temp"] > 35) & (df["humidity"] > 60),
        0.0004 * ((df["ambient_temp"] - 35) ** 1.5) * (df["humidity"] - 60),
        0
    )
    
    Tre = (
        CORE_TEMP_BASAL
        + CORE_METABOLIC_COEF * df["metabolic_rate"]
        + CORE_TEMP_AMB_COEF  * df["ambient_temp"]
        + CORE_HUMIDITY_COEF  * df["humidity"]
        - CORE_WIND_COEF      * df["wind_speed"]
        + CORE_RADIATION_COEF * df["solar_radiation"]
        - CORE_ACCL_COEF      * df["acclimatisation_days"]
        + CORE_BMI_COEF       * (df["bmi"] - 22.5)
        + CORE_WORKHOUR_COEF  * df["work_hours"]
        + CORE_AGE_COEF       * (df["age"] - 35)
        - CORE_HYDRATION_COEF * (df["hydration_level"] - 3)
        + interaction_penalty # Added non-linearity
    )
    return np.clip(Tre, 36.0, MAX_CORE_TEMP)

def compute_future_core_temperature(df: pd.DataFrame, current_tre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[FIX 1] Projects core temperature 30 minutes into the future based on trajectory."""
    # Heat accumulates based on work intensity and environment, mitigated by acclimatization
    base_increase = 0.10
    metabolic_load = (df["metabolic_rate"] - 100) * 0.0006
    env_load = np.maximum(0, df["ambient_temp"] - 32) * 0.015 + np.maximum(0, df["humidity"] - 50) * 0.005
    accl_protection = (df["acclimatisation_days"] / 90) * 0.15
    
    future_tre = current_tre + base_increase + metabolic_load + env_load - accl_protection
    future_tre += rng.normal(0, 0.05, size=len(df)) # Physiological variance
    return np.clip(future_tre, 36.0, MAX_CORE_TEMP)

def compute_heart_rate(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    HR = (
        HR_BASAL
        + HR_METABOLIC_COEF * df["metabolic_rate"]
        + HR_TEMP_COEF      * (df["ambient_temp"] - 28)    
        + HR_HUMIDITY_COEF  * (df["humidity"] - 50)         
        + HR_BMI_COEF       * (df["bmi"] - 22.5)
        + HR_AGE_COEF       * (df["age"] - 35)
    )
    HR += rng.normal(0, 4, size=len(df))
    return np.clip(HR, 50, 185)

def compute_sweat_rate(df: pd.DataFrame) -> np.ndarray:
    SR = (SWEAT_METABOLIC_COEF * df["metabolic_rate"] + SWEAT_TEMP_COEF * df["ambient_temp"] + SWEAT_HUMIDITY_COEF * df["humidity"])
    return np.clip(SR, 0, 1200)

def inject_gaussian_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    noisy = df.copy()
    for col, sigma in NOISE_SIGMA.items():
        if col in noisy.columns:
            noisy[col] = noisy[col] + rng.normal(0, sigma, size=len(noisy))
    noisy["ambient_temp"]  = noisy["ambient_temp"].clip(10, 55)
    noisy["humidity"]      = noisy["humidity"].clip(10, 100)
    noisy["heart_rate"]    = noisy["heart_rate"].clip(45, 190)
    noisy["wind_speed"]    = noisy["wind_speed"].clip(0, 10)
    return noisy

# ─────────────────────────────────────────────────────────────────────────────
# 3. LAG FEATURE CONSTRUCTION [FIX 3: DYNAMIC LAGS]
# ─────────────────────────────────────────────────────────────────────────────
def build_lag_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    out = df.copy()

    # Env lags: mostly stable, slight warming trend outdoors
    out["ambient_temp_t15"] = (df["ambient_temp"] - rng.normal(0.5, 0.3, size=n) + rng.normal(0, NOISE_SIGMA["ambient_temp"], size=n)).clip(10, 55)
    out["ambient_temp_t30"] = (df["ambient_temp"] - rng.normal(1.0, 0.5, size=n) + rng.normal(0, NOISE_SIGMA["ambient_temp"], size=n)).clip(10, 55)
    out["humidity_t15"] = (df["humidity"] - rng.normal(0.5, 1.0, size=n) + rng.normal(0, NOISE_SIGMA["humidity"], size=n)).clip(10, 100)
    out["humidity_t30"] = (df["humidity"] - rng.normal(1.0, 1.5, size=n) + rng.normal(0, NOISE_SIGMA["humidity"], size=n)).clip(10, 100)

    # [FIX 3] Dynamic HR Lags based on work_hours and acclimatisation
    # If work_hours < 0.5, they just started working, HR 30 mins ago was resting HR (~75)
    # If work_hours > 1.0, they were already working, HR drift is smaller.
    is_new_shift = (df["work_hours"] < 0.5).astype(int)
    accl_penalty = 1.0 - (df["acclimatisation_days"] / 90) # Less acclimated = steeper HR rise
    
    # Drift is large if new shift, driven by metabolism and acclimation otherwise
    hr_drift_t30 = np.where(
        is_new_shift == 1,
        df["heart_rate"] - (HR_BASAL + rng.normal(0, 5, size=n)), # Drops to resting
        (rng.normal(6.0, 2.0, size=n) * accl_penalty * (df["metabolic_rate"]/200)) # Gradual climb
    )
    hr_drift_t15 = hr_drift_t30 * 0.55 # Halfway point
    
    out["heart_rate_t30"] = (df["heart_rate"] - hr_drift_t30 + rng.normal(0, NOISE_SIGMA["heart_rate"], size=n)).clip(45, 190)
    out["heart_rate_t15"] = (df["heart_rate"] - hr_drift_t15 + rng.normal(0, NOISE_SIGMA["heart_rate"], size=n)).clip(45, 190)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# 4. RISK LABELLING [FIX 1: LABELING BASED ON FUTURE TRE]
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk_labels(Tre_future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conditions = [Tre_future < 37.5, (Tre_future >= 37.5) & (Tre_future < 38.0), (Tre_future >= 38.0) & (Tre_future < 38.5), Tre_future >= 38.5]
    risk_str = np.select(conditions, RISK_LABELS, default="Low")
    risk_num = np.select(conditions, [0, 1, 2, 3], default=0).astype(int)
    return risk_str, risk_num

def ensure_minimum_class_representation(df: pd.DataFrame, rng: np.random.Generator, min_pct: float = 0.08) -> pd.DataFrame:
    n_total = len(df)
    min_count = int(min_pct * n_total)
    augmented_chunks = [df]
    
    for label, label_num in RISK_LABEL_MAP.items():
        current = (df["risk_label_num"] == label_num).sum()
        deficit = max(0, min_count - current)
        if deficit > 0:
            extra = _generate_targeted_class(label, deficit, rng)
            augmented_chunks.append(extra)

    return pd.concat(augmented_chunks, ignore_index=True).sample(frac=1, random_state=rng.integers(1e6)).reset_index(drop=True)

def _generate_targeted_class(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    label_ranges = {
        "Low":      {"ambient_temp": (28, 34), "humidity": (30, 55), "metabolic_rate": (100, 200), "acclimatisation_days": (30, 90), "work_hours": (0, 4), "hydration_level_choices": [3, 4, 5]},
        "Moderate": {"ambient_temp": (34, 40), "humidity": (50, 70), "metabolic_rate": (180, 280), "acclimatisation_days": (15, 60), "work_hours": (2, 6), "hydration_level_choices": [2, 3, 4]},
        "High":     {"ambient_temp": (40, 45), "humidity": (65, 85), "metabolic_rate": (250, 350), "acclimatisation_days": (5, 30),  "work_hours": (4, 7), "hydration_level_choices": [1, 2, 3]},
        "Critical": {"ambient_temp": (43, 48), "humidity": (75, 95), "metabolic_rate": (300, 400), "acclimatisation_days": (0, 15),  "work_hours": (5, 8), "hydration_level_choices": [1, 2]},
    }
    r = label_ranges[label]
    attempts = n * 5
    params = {
        "age":                rng.integers(18, 61, size=attempts).astype(float),
        "bmi":                np.clip(rng.normal(24.5, 4, size=attempts), 17, 40),
        "acclimatisation_days": rng.uniform(*r["acclimatisation_days"][:2], size=attempts).round(),
        "ambient_temp":       rng.uniform(*r["ambient_temp"], size=attempts),
        "humidity":           rng.uniform(*r["humidity"], size=attempts),
        "wind_speed":         np.clip(rng.exponential(1.5, size=attempts), 0, 8),
        "solar_radiation":    rng.uniform(50, 1000, size=attempts),
        "metabolic_rate":     rng.uniform(*r["metabolic_rate"], size=attempts),
        "work_hours":         rng.uniform(*r["work_hours"], size=attempts),
        "hydration_level":    rng.choice(r["hydration_level_choices"], size=attempts).astype(float),
    }
    df_tmp = pd.DataFrame(params)
    Tre_current = compute_phs_core_temperature(df_tmp)
    Tre_future = compute_future_core_temperature(df_tmp, Tre_current, rng) # [FIX 1] Match target logic
    
    risk_str, risk_num = assign_risk_labels(Tre_future)
    df_tmp["core_temp_tre"] = Tre_current
    df_tmp["core_temp_tre_future"] = Tre_future
    df_tmp["risk_label_str"] = risk_str
    df_tmp["risk_label_num"] = risk_num
    df_tmp["sweat_rate"]    = compute_sweat_rate(df_tmp)
    df_tmp["heart_rate"]    = compute_heart_rate(df_tmp, rng)

    target_rows = df_tmp[df_tmp["risk_label_num"] == RISK_LABEL_MAP[label]]
    if len(target_rows) < n:
        target_rows = target_rows.sample(n=n, replace=True, random_state=int(rng.integers(1e6)))
    else:
        target_rows = target_rows.sample(n=n, random_state=int(rng.integers(1e6)))
    return target_rows.reset_index(drop=True)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    T, RH = out["ambient_temp"], out["humidity"]
    out["heat_index"] = (-8.784695 + 1.61139411*T + 2.338549*RH - 0.14611605*T*RH - 0.012308094*T**2 - 0.016424828*RH**2 + 0.002211732*T**2*RH + 0.00072546*T*RH**2 - 0.000003582*T**2*RH**2).clip(25, 70)
    out["hr_delta_t15"] = out["heart_rate"] - out["heart_rate_t15"]
    out["hr_delta_t30"] = out["heart_rate"] - out["heart_rate_t30"]
    out["temp_delta_t15"] = out["ambient_temp"] - out["ambient_temp_t15"]
    out["temp_humidity_product"] = (out["ambient_temp"] * out["humidity"]) / 100.0
    return out

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    personal    = ["age", "bmi", "acclimatisation_days"]
    env_current = ["ambient_temp", "humidity", "wind_speed", "solar_radiation"]
    env_lag     = ["ambient_temp_t15", "ambient_temp_t30", "humidity_t15", "humidity_t30"]
    work        = ["metabolic_rate", "work_hours", "hydration_level"]
    physio      = ["heart_rate", "heart_rate_t15", "heart_rate_t30", "sweat_rate", "core_temp_tre", "core_temp_tre_future"]
    derived     = ["heat_index", "hr_delta_t15", "hr_delta_t30", "temp_delta_t15", "temp_humidity_product"]
    labels      = ["risk_label_str", "risk_label_num"]
    ordered = personal + env_current + env_lag + work + physio + derived + labels
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]

def export_feature_dictionary(df: pd.DataFrame, output_dir: str):
    # Truncated dictionary for brevity, append standard definitions
    pass 

def export_summary_report(df: pd.DataFrame, output_dir: str, n_requested: int, seed: int, elapsed: float):
    pass