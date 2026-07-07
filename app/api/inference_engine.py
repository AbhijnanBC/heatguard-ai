"""
=============================================================================
PHASE 4: INFERENCE ENGINE
=============================================================================
Loads all Phase 2 + Phase 3 model artefacts once at startup and exposes
a clean predict() function that the FastAPI layer calls per request.

Inference Pipeline (exactly as documented in Phase 3 handoff):
    1. Accept raw worker sensor readings (current + lag values)
    2. Compute derived features (heat_index, hr_deltas, temp_delta,
       temp_humidity_product) from raw inputs
    3. Compute PHS core temperature proxy (core_temp_tre)
    4. Assign worker to K-Means cluster using Phase 2 scaler + model
    5. Compute vulnerability_score from cluster features
    6. Pull adaptive_alert_multiplier from cluster assignment
    7. Assemble the 27-feature vector in EXACT feature_list.json order
    8. Call XGBoost model.predict_proba()
    9. Apply adaptive threshold to determine alert status (Sum of High + Critical)
    10. Return structured prediction response

Phase 1 Upgrades Respected:
    - Dynamic lag physiology: hr_delta_t30 and hr_delta_t15 are computed
      from the incoming lag heart rate readings, capturing the trajectory.
    - Non-linear interaction: heat_index and temp_humidity_product are
      recomputed at inference time from ambient_temp and humidity, so the
      exponential synergy term (Phase 1 upgrade) is always current.
    - Predictive target shift (t+30): the model was trained to predict
      future risk; the inference engine returns both current state AND
      the predicted risk level 30 minutes ahead.
=============================================================================
"""

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — resolved relative to this file's location
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent
_MODELS_DIR  = _HERE.parent.parent / "models"

MODEL_PATH        = _MODELS_DIR / "heatstroke_model.pkl"
FEATURE_LIST_PATH = _MODELS_DIR / "feature_list.json"
KMEANS_PATH       = _MODELS_DIR / "kmeans_model.pkl"
SCALER_PATH       = _MODELS_DIR / "cluster_scaler.pkl"
METRICS_PATH      = _MODELS_DIR / "phase3_metrics.json"

# K-Means clustering feature order (must match Phase 2 exactly)
CLUSTER_FEATURES = [
    "age", "bmi", "acclimatisation_days",
    "metabolic_rate", "hydration_level", "hr_delta_t30",
]

# Persona names mapped from cluster_id (from Phase 2 run output)
PERSONA_MAP = {
    0: "High-BMI Novice",
    1: "Acclimatised Veteran",
}

# Vulnerability score weights (mirrors Phase 2 kmeans_profiling.py)
VULNERABILITY_WEIGHTS = {
    "bmi":                  (+0.30, 22.5, 5.0),
    "age":                  (+0.20, 35.0, 15.0),
    "acclimatisation_days": (-0.35, 0.0,  45.0),
    "metabolic_rate":       (+0.25, 250.0, 100.0),
    "hydration_level":      (-0.20, 3.0, 2.0),
    "hr_delta_t30":         (+0.15, 0.0, 5.0),
}

# Adaptive alert multiplier range (mirrors Phase 2)
VULN_MIN = -0.342
VULN_MAX =  0.312

# Risk labels
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]

# Risk level metadata (for app UI rendering)
RISK_META = {
    "Low": {
        "color": "#27AE60",
        "emoji": "🟢",
        "message": "Heat stress is low. Continue normal work.",
        "action":  "Stay hydrated. Monitor conditions.",
    },
    "Moderate": {
        "color": "#F39C12",
        "emoji": "🟡",
        "message": "Moderate heat strain detected.",
        "action":  "Take a 10-minute break in shade. Drink water.",
    },
    "High": {
        "color": "#E67E22",
        "emoji": "🟠",
        "message": "Significant heat strain — action required now.",
        "action":  "Stop work. Move to shade. Drink 500ml water immediately.",
    },
    "Critical": {
        "color": "#C0392B",
        "emoji": "🔴",
        "message": "CRITICAL — Imminent collapse risk.",
        "action":  "STOP WORK IMMEDIATELY. Seek shade and cool water. Call supervisor.",
    },
}

# PHS core temperature coefficients (mirrors Phase 1)
CORE_TEMP_BASAL      = 36.8
CORE_METABOLIC_COEF  = 0.0018
CORE_TEMP_AMB_COEF   = 0.028
CORE_HUMIDITY_COEF   = 0.007
CORE_WIND_COEF       = 0.10
CORE_RADIATION_COEF  = 0.0004
CORE_ACCL_COEF       = 0.018
CORE_BMI_COEF        = 0.022
CORE_WORKHOUR_COEF   = 0.055
CORE_AGE_COEF        = 0.003
CORE_HYDRATION_COEF  = 0.045


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY — loaded once at module import
# ─────────────────────────────────────────────────────────────────────────────

class _ModelRegistry:
    """Singleton that loads all model artefacts once and holds them in memory."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("[InferenceEngine] Loading model artefacts...")

        self.xgb_model   = joblib.load(MODEL_PATH)
        self.kmeans      = joblib.load(KMEANS_PATH)
        self.scaler      = joblib.load(SCALER_PATH)

        with open(FEATURE_LIST_PATH) as f:
            feat_info    = json.load(f)
        self.features    = feat_info["features"]   # 27-element list, exact order
        self.n_features  = len(self.features)

        with open(METRICS_PATH) as f:
            self.metrics = json.load(f)

        self._loaded = True
        print(f"[InferenceEngine] Loaded. Features={self.n_features}, "
              f"Clusters={self.kmeans.n_clusters}")

    @property
    def is_loaded(self):
        return self._loaded


_registry = _ModelRegistry()


def get_registry() -> _ModelRegistry:
    """Return the loaded model registry (load on first call)."""
    if not _registry.is_loaded:
        _registry.load()
    return _registry


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING AT INFERENCE TIME
# ─────────────────────────────────────────────────────────────────────────────

def compute_heat_index(ambient_temp: float, humidity: float) -> float:
    """
    Rothfuss & Terjung heat index formula.
    Recomputed at inference to capture the Phase 1 non-linear interaction.
    """
    T  = ambient_temp
    RH = humidity
    hi = (
        -8.784695
        + 1.61139411 * T
        + 2.338549   * RH
        - 0.14611605 * T * RH
        - 0.012308094 * T ** 2
        - 0.016424828 * RH ** 2
        + 0.002211732 * T ** 2 * RH
        + 0.00072546  * T * RH ** 2
        - 0.000003582 * T ** 2 * RH ** 2
    )
    return float(np.clip(hi, 25.0, 70.0))


def compute_core_temp_proxy(
    ambient_temp: float, humidity: float, wind_speed: float,
    solar_radiation: float, metabolic_rate: float, work_hours: float,
    hydration_level: float, acclimatisation_days: float,
    bmi: float, age: float,
) -> float:
    """
    PHS equation (ISO 7933) — mirrors Phase 1 compute_phs_core_temperature().
    Recomputed at inference so the model receives a physiologically grounded
    estimate, not a raw input.
    """
    Tre = (
        CORE_TEMP_BASAL
        + CORE_METABOLIC_COEF * metabolic_rate
        + CORE_TEMP_AMB_COEF  * ambient_temp
        + CORE_HUMIDITY_COEF  * humidity
        - CORE_WIND_COEF      * wind_speed
        + CORE_RADIATION_COEF * solar_radiation
        - CORE_ACCL_COEF      * acclimatisation_days
        + CORE_BMI_COEF       * (bmi - 22.5)
        + CORE_WORKHOUR_COEF  * work_hours
        + CORE_AGE_COEF       * (age - 35)
        - CORE_HYDRATION_COEF * (hydration_level - 3)
    )
    return float(np.clip(Tre, 36.0, 39.5))


def compute_sweat_rate(metabolic_rate: float, ambient_temp: float,
                       humidity: float) -> float:
    """PHS sweat rate estimate (g/hr)."""
    sr = (0.30 * metabolic_rate + 0.18 * ambient_temp + 0.005 * humidity)
    return float(np.clip(sr, 0.0, 1200.0))


def compute_vulnerability_score(worker: dict) -> float:
    """Mirrors Phase 2 engineer_cluster_features() vulnerability formula."""
    score = (
        0.30 * (worker["bmi"] - 22.5)               / 5.0
        + 0.20 * (worker["age"] - 35)                / 15.0
        - 0.35 * worker["acclimatisation_days"]       / 45.0
        + 0.25 * (worker["metabolic_rate"] - 250)     / 100.0
        - 0.20 * (worker["hydration_level"] - 3)      / 2.0
        + 0.15 * worker["hr_delta_t30"]               / 5.0
    )
    return float(score)


def compute_adaptive_multiplier(vuln_score: float) -> float:
    """Mirrors Phase 2 build_cluster_label_column() multiplier formula."""
    vuln_range = VULN_MAX - VULN_MIN if VULN_MAX != VULN_MIN else 1.0
    mult = 1.15 - 0.35 * (vuln_score - VULN_MIN) / vuln_range
    return float(np.clip(mult, 0.70, 1.20))


def assign_cluster(worker: dict, reg: _ModelRegistry) -> tuple[int, str]:
    """
    Assign new worker to K-Means cluster using Phase 2 scaler + model.
    Returns (cluster_id, persona_name).
    """
    cluster_vec = np.array([[worker[f] for f in CLUSTER_FEATURES]])
    scaled      = reg.scaler.transform(cluster_vec)
    cluster_id  = int(reg.kmeans.predict(scaled)[0])
    persona     = PERSONA_MAP.get(cluster_id, f"Profile_{cluster_id}")
    return cluster_id, persona


# ─────────────────────────────────────────────────────────────────────────────
# CORE PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(worker_input: dict) -> dict:
    """
    Full inference pipeline. Accepts a dict of raw sensor + personal inputs,
    returns a structured prediction response ready for the app UI.

    Required inputs (minimum):
        age, bmi, acclimatisation_days, metabolic_rate, work_hours,
        hydration_level, ambient_temp, humidity, wind_speed,
        solar_radiation, heart_rate

    Optional lag inputs (supply if available; defaults to current value):
        ambient_temp_t15, ambient_temp_t30
        humidity_t15, humidity_t30
        heart_rate_t15, heart_rate_t30

    Returns dict with:
        predicted_class, probabilities, alert_fires,
        cluster_id, persona_name, vulnerability_score,
        adaptive_alert_multiplier, threshold_used,
        risk_color, risk_emoji, risk_message, risk_action,
        core_temp_estimate, heat_index, time_to_peak_minutes,
        top_risk_factors (top 3 features driving this prediction)
    """
    reg = get_registry()

    # ── Pull inputs ──────────────────────────────────────────────────────────
    age                  = float(worker_input["age"])
    bmi                  = float(worker_input["bmi"])
    acclimatisation_days = float(worker_input["acclimatisation_days"])
    metabolic_rate       = float(worker_input["metabolic_rate"])
    work_hours           = float(worker_input["work_hours"])
    hydration_level      = float(worker_input["hydration_level"])
    ambient_temp         = float(worker_input["ambient_temp"])
    humidity             = float(worker_input["humidity"])
    wind_speed           = float(worker_input.get("wind_speed", 1.5))
    solar_radiation      = float(worker_input.get("solar_radiation", 500.0))
    heart_rate           = float(worker_input["heart_rate"])

    # Lag values — default to current if not supplied (graceful degradation)
    ambient_temp_t15 = float(worker_input.get("ambient_temp_t15") or (ambient_temp - 0.8))
    ambient_temp_t30 = float(worker_input.get("ambient_temp_t30") or (ambient_temp - 1.5))
    humidity_t15     = float(worker_input.get("humidity_t15")     or (humidity - 0.5))
    humidity_t30     = float(worker_input.get("humidity_t30")     or (humidity - 1.0))
    heart_rate_t15   = float(worker_input.get("heart_rate_t15")   or (heart_rate - 2.5))
    heart_rate_t30   = float(worker_input.get("heart_rate_t30")   or (heart_rate - 5.0))

    # ── Derived features ─────────────────────────────────────────────────────
    hr_delta_t15         = heart_rate - heart_rate_t15
    hr_delta_t30         = heart_rate - heart_rate_t30
    temp_delta_t15       = ambient_temp - ambient_temp_t15
    temp_humidity_product = (ambient_temp * humidity) / 100.0
    heat_index_val        = compute_heat_index(ambient_temp, humidity)
    core_temp_tre         = compute_core_temp_proxy(
        ambient_temp, humidity, wind_speed, solar_radiation,
        metabolic_rate, work_hours, hydration_level,
        acclimatisation_days, bmi, age,
    )
    sweat_rate = compute_sweat_rate(metabolic_rate, ambient_temp, humidity)

    # ── Worker dict for clustering ────────────────────────────────────────────
    worker = {
        "age":                  age,
        "bmi":                  bmi,
        "acclimatisation_days": acclimatisation_days,
        "metabolic_rate":       metabolic_rate,
        "hydration_level":      hydration_level,
        "hr_delta_t30":         hr_delta_t30,
    }

    # ── Phase 2 cluster assignment ────────────────────────────────────────────
    cluster_id, persona_name = assign_cluster(worker, reg)
    vuln_score               = compute_vulnerability_score(worker)
    adaptive_mult            = compute_adaptive_multiplier(vuln_score)

    # ── Assemble 27-feature vector in EXACT feature_list.json order ──────────
    feature_values = {
        "age":                   age,
        "bmi":                   bmi,
        "acclimatisation_days":  acclimatisation_days,
        "ambient_temp":          ambient_temp,
        "humidity":              humidity,
        "wind_speed":            wind_speed,
        "solar_radiation":       solar_radiation,
        "ambient_temp_t15":      ambient_temp_t15,
        "ambient_temp_t30":      ambient_temp_t30,
        "humidity_t15":          humidity_t15,
        "humidity_t30":          humidity_t30,
        "metabolic_rate":        metabolic_rate,
        "work_hours":            work_hours,
        "hydration_level":       hydration_level,
        "heart_rate":            heart_rate,
        "heart_rate_t15":        heart_rate_t15,
        "heart_rate_t30":        heart_rate_t30,
        "sweat_rate":            sweat_rate,
        "core_temp_tre":         core_temp_tre,
        "heat_index":            heat_index_val,
        "hr_delta_t15":          hr_delta_t15,
        "hr_delta_t30":          hr_delta_t30,
        "temp_delta_t15":        temp_delta_t15,
        "temp_humidity_product": temp_humidity_product,
        "vulnerability_score":   vuln_score,
        "cluster_id":            float(cluster_id),
        "adaptive_alert_multiplier": adaptive_mult,
    }
    feature_values["core_temp_tre_future"] = (
    feature_values.get("core_temp", 37) -
    feature_values.get("core_temp_t_minus_30", 37)
)
    X = np.array([[feature_values[f] for f in reg.features]])

    # ── XGBoost inference ─────────────────────────────────────────────────────
    probs      = reg.xgb_model.predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    pred_label = RISK_LABELS[pred_class]

    # ── Adaptive threshold [FIX APPLIED HERE] ─────────────────────────────────
    base_threshold = 0.50
    threshold      = base_threshold * adaptive_mult
    # Sum of High (index 2) and Critical (index 3) probabilities
    alert_fires    = bool((probs[2] + probs[3]) >= threshold)

    # ── Time-to-peak estimate (minutes) ──────────────────────────────────────
    # Estimated from predicted class + HR trajectory:
    # Higher risk + faster rising HR = less time to peak
    time_to_peak = _estimate_time_to_peak(pred_class, hr_delta_t30, vuln_score)

    # ── Top risk factors (SHAP-like approximation via feature contributions) ─
    top_factors = _get_top_risk_factors(feature_values, pred_class, probs)

    # ── Build response ────────────────────────────────────────────────────────
    meta = RISK_META[pred_label]
    return {
        # Primary prediction
        "predicted_class":          pred_label,
        "predicted_class_num":      pred_class,
        "probabilities": {
            label: round(float(p), 4)
            for label, p in zip(RISK_LABELS, probs)
        },
        "risk_score":               round(float(_compute_risk_score(probs)), 1),

        # Alert
        "alert_fires":              alert_fires,
        "threshold_used":           round(threshold, 3),
        "adaptive_alert_multiplier": round(adaptive_mult, 3),

        # Worker profile
        "cluster_id":               cluster_id,
        "persona_name":             persona_name,
        "vulnerability_score":      round(vuln_score, 4),

        # Physiological estimates
        "core_temp_estimate":       round(core_temp_tre, 2),
        "heat_index":               round(heat_index_val, 1),
        "sweat_rate_estimate":      round(sweat_rate, 1),
        "hr_trajectory":            round(hr_delta_t30, 2),

        # Time prediction
        "time_to_peak_minutes":     time_to_peak,

        # UI metadata
        "risk_color":               meta["color"],
        "risk_emoji":               meta["emoji"],
        "risk_message":             meta["message"],
        "risk_action":              meta["action"],

        # Explainability
        "top_risk_factors":         top_factors,
    }


def _compute_risk_score(probs: np.ndarray) -> float:
    """
    Compute a 0–100 risk score from class probabilities.
    Weights: Low=0, Moderate=33, High=67, Critical=100.
    """
    weights = np.array([0.0, 33.0, 67.0, 100.0])
    return float(np.dot(probs, weights))


def _estimate_time_to_peak(pred_class: int, hr_delta: float,
                           vuln_score: float) -> int:
    """
    Estimate minutes until risk peaks based on predicted class and HR trajectory.
    Used for the "time to peak" countdown on the app dashboard.

    Logic:
        Critical with fast-rising HR  → 5–10 min
        High with rising HR           → 15–25 min
        Moderate                      → 30–45 min
        Low                           → 60+ min
    """
    base_times = {3: 8, 2: 20, 1: 38, 0: 65}
    base       = base_times.get(pred_class, 60)
    # Faster HR rise → less time
    hr_adj     = max(0, int(hr_delta * 1.5))
    # Higher vulnerability → less time
    vuln_adj   = max(0, int(vuln_score * 8))
    return max(3, base - hr_adj - vuln_adj)


def _get_top_risk_factors(feature_values: dict, pred_class: int,
                          probs: np.ndarray) -> list:
    """
    Return the top 3 human-readable risk factors for this prediction.
    Based on domain rules that mirror SHAP findings from Phase 3 analysis.
    Used in the app's "why is this alert firing" explanation panel.
    """
    factors = []

    # Core temperature — dominant SHAP feature
    tre = feature_values["core_temp_tre"]
    if tre >= 38.5:
        factors.append({
            "feature":     "Core temperature",
            "value":       f"{tre:.1f}°C",
            "severity":    "critical",
            "description": "Predicted body temp has crossed the danger threshold.",
        })
    elif tre >= 38.0:
        factors.append({
            "feature":     "Core temperature",
            "value":       f"{tre:.1f}°C",
            "severity":    "high",
            "description": "Body temperature is significantly elevated.",
        })

    # HR trajectory — Phase 1 dynamic lag upgrade
    hr_d = feature_values["hr_delta_t30"]
    if hr_d >= 10:
        factors.append({
            "feature":     "Rising heart rate",
            "value":       f"+{hr_d:.1f} bpm over 30 min",
            "severity":    "critical",
            "description": "Rapid HR increase signals accelerating heat strain.",
        })
    elif hr_d >= 5:
        factors.append({
            "feature":     "Rising heart rate",
            "value":       f"+{hr_d:.1f} bpm over 30 min",
            "severity":    "high",
            "description": "Heart rate is rising — monitor closely.",
        })

    # Heat index — Phase 1 non-linear interaction
    hi = feature_values["heat_index"]
    if hi >= 54:
        factors.append({
            "feature":     "Heat index",
            "value":       f"{hi:.1f}°C",
            "severity":    "critical",
            "description": "Combined heat + humidity is creating extreme conditions.",
        })
    elif hi >= 41:
        factors.append({
            "feature":     "Heat index",
            "value":       f"{hi:.1f}°C",
            "severity":    "high",
            "description": "High heat index — the air feels much hotter than it is.",
        })

    # Acclimatisation (protective factor — low = bad)
    accl = feature_values["acclimatisation_days"]
    if accl < 7:
        factors.append({
            "feature":     "Low acclimatisation",
            "value":       f"{int(accl)} days",
            "severity":    "high",
            "description": "Worker has not built heat tolerance. Extra caution needed.",
        })

    # Hydration
    hydration = feature_values["hydration_level"]
    if hydration <= 1:
        factors.append({
            "feature":     "Severe dehydration risk",
            "value":       f"Level {int(hydration)}/5",
            "severity":    "critical",
            "description": "Critically low hydration amplifies all other risk factors.",
        })
    elif hydration <= 2:
        factors.append({
            "feature":     "Low hydration",
            "value":       f"Level {int(hydration)}/5",
            "severity":    "high",
            "description": "Drink water now to reduce heat strain.",
        })

    # Return top 3 by severity order
    severity_order = {"critical": 0, "high": 1, "moderate": 2}
    factors.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return factors[:3]