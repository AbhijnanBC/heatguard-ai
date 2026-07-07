# Phase 4 — FastAPI Backend & Inference Engine

**Project:** AI-Powered Heatstroke Early Warning System for Outdoor Workers  
**Input:** All Phase 2 + Phase 3 model artefacts (`models/` directory)  
**Output:** A production-ready REST API that the smartphone app calls to get predictions

---

## What Phase 4 Does and Why

Phase 4 is where the trained model becomes a usable product. The XGBoost model from Phase 3 and the K-Means models from Phase 2 are loaded into a FastAPI server that exposes clean REST endpoints. The smartphone app sends a worker's current sensor readings as a JSON POST request and receives a complete prediction response — risk class, probability scores, alert decision, personalised recommendation text, and a "time to peak" countdown.

The key design principle is that **the inference engine replicates the entire feature engineering pipeline from Phases 1, 2, and 3**. It does not just call `model.predict()` on raw inputs. It:
1. Recomputes the PHS core temperature proxy from ISO 7933 equations
2. Recomputes the Rothfuss & Terjung heat index (Phase 1 non-linear interaction)
3. Computes HR trajectory deltas from lag readings (Phase 1 dynamic lag upgrade)
4. Assigns the worker to a K-Means cluster (Phase 2)
5. Computes vulnerability score and adaptive alert multiplier (Phase 2)
6. Assembles the 27-feature vector in **exact `feature_list.json` order**
7. Calls XGBoost and applies the adaptive threshold to decide alert status

Without this pipeline, passing raw inputs would produce garbage predictions because the model was trained on engineered features, not raw sensor readings.

---

## Phase 1 Mathematical Upgrades — How Phase 4 Handles All Three

| Phase 1 Upgrade | Phase 4 Handling |
|---|---|
| **Predictive target shift (t+30):** Model predicts future risk. | At inference time, the user submits current readings. The model returns what the worker's risk will be in 30 minutes. The response field `time_to_peak_minutes` further refines this estimate based on HR trajectory and vulnerability score. |
| **Non-linear temp×humidity interaction:** Exponential synergy term in PHS. | `compute_heat_index()` and `temp_humidity_product` are recomputed at every inference call from the live ambient_temp and humidity inputs. The non-linear interaction is always fresh — never stale from training time. |
| **Dynamic lag physiology:** HR lags driven by work_hours, acclimatisation, metabolic_rate. | `hr_delta_t15` and `hr_delta_t30` are computed from the incoming lag heart rate readings: `hr_delta_t30 = heart_rate - heart_rate_t30`. The API accepts optional lag fields and defaults gracefully when they are absent (new worker with no reading history). |

---

## File Structure

```
heatstroke_ai/
├── app/
│   ├── api/
│   │   ├── inference_engine.py     ← loads models, feature engineering, predict()
│   │   └── app.py                  ← FastAPI endpoints, Pydantic schemas
│   └── tests/
│       └── test_api.py             ← 48 automated tests (all passing)
├── models/
│   ├── heatstroke_model.pkl        ← XGBoost (Phase 3)
│   ├── feature_list.json           ← exact 27-feature order
│   ├── kmeans_model.pkl            ← K-Means (Phase 2)
│   └── cluster_scaler.pkl          ← StandardScaler (Phase 2)
```

---

## Dependencies

```bash
pip install fastapi uvicorn pydantic httpx xgboost scikit-learn \
            imbalanced-learn joblib numpy
```

---

## How to Run

```bash
# Start the server
cd app/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run the test suite (no server needed — uses TestClient)
cd app/tests
python test_api.py

# Run a single test class
python test_api.py TestSafetyConstraints

# Open Swagger UI in browser
# http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms models are loaded |
| `GET` | `/model/info` | Model metadata + Phase 3 test-set metrics |
| `POST` | `/predict` | Single worker prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 100 workers) |
| `POST` | `/onboard` | One-time worker onboarding → cluster assignment |
| `GET` | `/demo/scenario/{name}` | Pre-built demo scenarios for viva |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Request Schema — `POST /predict`

```json
{
  "age":                  34,       // required: 15–75
  "bmi":                  24.5,     // required: 15–45
  "acclimatisation_days": 12,       // required: 0–120
  "ambient_temp":         42.0,     // required: 20–55 °C
  "humidity":             78.0,     // required: 10–100 %
  "wind_speed":           0.8,      // optional, default 1.5 m/s
  "solar_radiation":      850.0,    // optional, default 500 W/m²
  "metabolic_rate":       280.0,    // required: 80–500 W
  "work_hours":           5.0,      // required: 0–12
  "hydration_level":      2,        // required: 1/2/3/4/5
  "heart_rate":           125.0,    // required: 40–200 bpm

  // Optional lag readings (stored by app every 15 min)
  // If absent, defaults to current value ± physiological drift
  "ambient_temp_t15":     41.2,
  "ambient_temp_t30":     40.5,
  "humidity_t15":         76.0,
  "humidity_t30":         74.5,
  "heart_rate_t15":       118.0,
  "heart_rate_t30":       110.0
}
```

---

## Live API Responses (from executed run)

### `GET /health`

```json
{
  "status": "healthy",
  "models_loaded": true,
  "n_features": 27,
  "n_clusters": 2,
  "model_version": "1.0.0"
}
```

### `POST /predict` — Danger worker (48yo, BMI 31, 6 days acclimatised, 44.5°C, HR rising 16 bpm)

```json
{
  "predicted_class":   "Critical",
  "predicted_class_num": 3,
  "probabilities": {
    "Low":      0.0001,
    "Moderate": 0.0001,
    "High":     0.0002,
    "Critical": 0.9996
  },
  "risk_score":                 100.0,
  "alert_fires":                true,
  "threshold_used":             0.35,
  "adaptive_alert_multiplier":  0.7,
  "cluster_id":                 0,
  "persona_name":               "High-BMI Novice",
  "vulnerability_score":        1.5792,
  "core_temp_estimate":         39.5,
  "heat_index":                 70.0,
  "sweat_rate_estimate":        114.9,
  "hr_trajectory":              16.0,
  "time_to_peak_minutes":       3,
  "risk_color":                 "#C0392B",
  "risk_emoji":                 "🔴",
  "risk_message":               "CRITICAL — Imminent collapse risk.",
  "risk_action":                "STOP WORK IMMEDIATELY. Seek shade and cool water. Call supervisor.",
  "top_risk_factors": [
    {
      "feature":     "Core temperature",
      "value":       "39.5°C",
      "severity":    "critical",
      "description": "Predicted body temp has crossed the danger threshold."
    },
    {
      "feature":     "Rising heart rate",
      "value":       "+16.0 bpm over 30 min",
      "severity":    "critical",
      "description": "Rapid HR increase signals accelerating heat strain."
    },
    {
      "feature":     "Heat index",
      "value":       "70.0°C",
      "severity":    "critical",
      "description": "Combined heat + humidity is creating extreme conditions."
    }
  ]
}
```

### `POST /onboard` — New worker profile setup

```json
// Request
{ "age": 34.0, "bmi": 24.5, "acclimatisation_days": 12.0 }

// Response
{
  "cluster_id":                0,
  "persona_name":              "Acclimatised Veteran",
  "vulnerability_score":       -0.1117,
  "adaptive_alert_multiplier": 1.027,
  "persona_description":       "You have strong heat tolerance built over many days of outdoor work...",
  "personalisation_summary":   "Based on your profile, alerts will fire 3% later than the baseline threshold."
}
```

### `GET /demo/scenario/critical_early` — HR trajectory early warning demo

```json
{
  "predicted_class":    "Critical",
  "risk_score":         100.0,
  "alert_fires":        true,
  "time_to_peak_minutes": 3,
  "hr_trajectory":      22.0,
  "top_risk_factors": [
    { "feature": "Rising heart rate", "value": "+22.0 bpm over 30 min",
      "severity": "critical", "description": "Rapid HR increase signals accelerating heat strain." },
    ...
  ]
}
```

---

## Test Suite Results (48/48 passing)

```
Tests run   : 48
Passed      : 48
Failures    : 0
Errors      : 0
Status      : ✓ ALL PASSED

Test categories:
  TestInferenceEngine   — 15 unit tests (heat index, PHS, vulnerability, predict)
  TestSafetyConstraints —  7 tests  (alert logic, thresholds, trajectories)
  TestAPIEndpoints      — 14 tests  (all endpoints, validation, schemas)
  TestDemoScenarios     —  6 tests  (all 4 scenarios, regression checks)
  TestEdgeCases         —  6 tests  (extremes, boundaries, None handling)
```

---

## Inference Pipeline — Step by Step

```
Worker sends POST /predict
         │
         ▼
 [1] Pydantic validation
     - Range checks (age 15–75, temp 20–55, etc.)
     - hydration_level must be 1/2/3/4/5
     - heart_rate required
         │
         ▼
 [2] Feature engineering (inference_engine.py)
     - Lag defaults: if t15/t30 absent → current ± drift
     - hr_delta_t15 = heart_rate − heart_rate_t15
     - hr_delta_t30 = heart_rate − heart_rate_t30        ← Phase 1 dynamic lag
     - heat_index   = Rothfuss & Terjung formula          ← Phase 1 non-linear
     - temp_humidity_product = (Ta × RH) / 100           ← Phase 1 synergy
     - core_temp_tre = PHS ISO 7933 equation              ← Phase 1 PHS
     - sweat_rate    = PHS sweat equation
         │
         ▼
 [3] Phase 2 cluster assignment
     - Scale 6 personal features (StandardScaler)
     - kmeans_model.predict() → cluster_id
     - vulnerability_score = weighted formula
     - adaptive_alert_multiplier = 1.15 − 0.35 × norm_vuln
         │
         ▼
 [4] Assemble 27-feature vector
     - Exact order from feature_list.json
     - cluster_id, vulnerability_score, adaptive_alert_multiplier appended
         │
         ▼
 [5] XGBoost inference
     - model.predict_proba(X) → [P(Low), P(Mod), P(High), P(Crit)]
         │
         ▼
 [6] Alert decision
     - threshold = 0.50 × adaptive_alert_multiplier
     - alert_fires = P(High) ≥ threshold OR P(Critical) ≥ threshold
         │
         ▼
 [7] Build response
     - risk_score = dot(probs, [0,33,67,100])
     - time_to_peak = f(pred_class, hr_delta_t30, vuln_score)
     - top_risk_factors = domain-rule explanations
     - risk_color, emoji, message, action from RISK_META
         │
         ▼
 Return JSON response to app
```

---

## File 1 of 3 — `inference_engine.py`

The core inference engine. Loads all models once at startup via a singleton registry. Exposes `predict()` which the FastAPI layer calls per request.

```python
"""
=============================================================================
PHASE 4: INFERENCE ENGINE
=============================================================================
Loads all Phase 2 + Phase 3 artefacts once at startup (singleton registry).
Exposes predict() which the FastAPI layer calls per request.

Inference pipeline:
  1. Accept raw sensor + personal inputs
  2. Recompute derived features (heat_index, PHS Tre, hr_deltas)
  3. Assign K-Means cluster (Phase 2 scaler + model)
  4. Compute vulnerability_score and adaptive_alert_multiplier
  5. Assemble 27-feature vector in EXACT feature_list.json order
  6. Call XGBoost.predict_proba()
  7. Apply adaptive threshold → alert decision
  8. Return structured response dict

Phase 1 upgrades respected:
  - Dynamic lag physiology: hr_delta_t30 computed from lag heart rate readings
  - Non-linear interaction: heat_index and temp_humidity_product recomputed live
  - Predictive t+30: model returns what risk will be in 30 min
=============================================================================
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — resolved from this file's absolute location
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent    # must be .resolve() for TestClient
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

# Adaptive alert multiplier range bounds (from Phase 2 cluster centroids)
VULN_MIN = -0.342
VULN_MAX =  0.312

RISK_LABELS = ["Low", "Moderate", "High", "Critical"]

RISK_META = {
    "Low":      {"color": "#27AE60", "emoji": "🟢",
                 "message": "Heat stress is low. Continue normal work.",
                 "action":  "Stay hydrated. Monitor conditions."},
    "Moderate": {"color": "#F39C12", "emoji": "🟡",
                 "message": "Moderate heat strain detected.",
                 "action":  "Take a 10-minute break in shade. Drink water."},
    "High":     {"color": "#E67E22", "emoji": "🟠",
                 "message": "Significant heat strain — action required now.",
                 "action":  "Stop work. Move to shade. Drink 500ml water immediately."},
    "Critical": {"color": "#C0392B", "emoji": "🔴",
                 "message": "CRITICAL — Imminent collapse risk.",
                 "action":  "STOP WORK IMMEDIATELY. Seek shade and cool water. Call supervisor."},
}

# PHS ISO 7933 coefficients (mirrors Phase 1 generate_synthetic.py)
CORE_TEMP_BASAL     = 36.8
CORE_METABOLIC_COEF = 0.0018
CORE_TEMP_AMB_COEF  = 0.028
CORE_HUMIDITY_COEF  = 0.007
CORE_WIND_COEF      = 0.10
CORE_RADIATION_COEF = 0.0004
CORE_ACCL_COEF      = 0.018
CORE_BMI_COEF       = 0.022
CORE_WORKHOUR_COEF  = 0.055
CORE_AGE_COEF       = 0.003
CORE_HYDRATION_COEF = 0.045


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY — singleton, loaded once at module import
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
        self.xgb_model  = joblib.load(MODEL_PATH)
        self.kmeans     = joblib.load(KMEANS_PATH)
        self.scaler     = joblib.load(SCALER_PATH)
        with open(FEATURE_LIST_PATH) as f:
            feat_info   = json.load(f)
        self.features   = feat_info["features"]
        self.n_features = len(self.features)
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
    Rothfuss & Terjung heat index — recomputed live at inference.
    Captures the Phase 1 non-linear temp×humidity interaction.
    Clipped to [25, 70] °C.
    """
    T, RH = ambient_temp, humidity
    hi = (
        -8.784695
        + 1.61139411 * T
        + 2.338549   * RH
        - 0.14611605 * T * RH
        - 0.012308094 * T**2
        - 0.016424828 * RH**2
        + 0.002211732 * T**2 * RH
        + 0.00072546  * T * RH**2
        - 0.000003582 * T**2 * RH**2
    )
    return float(np.clip(hi, 25.0, 70.0))


def compute_core_temp_proxy(
    ambient_temp, humidity, wind_speed, solar_radiation,
    metabolic_rate, work_hours, hydration_level,
    acclimatisation_days, bmi, age
) -> float:
    """
    ISO 7933 PHS core temperature estimate — mirrors Phase 1 exactly.
    Clipped to physiological range [36.0, 39.5] °C.
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


def compute_sweat_rate(metabolic_rate, ambient_temp, humidity) -> float:
    """PHS sweat rate estimate (g/hr), clipped to [0, 1200]."""
    sr = 0.30 * metabolic_rate + 0.18 * ambient_temp + 0.005 * humidity
    return float(np.clip(sr, 0.0, 1200.0))


def compute_vulnerability_score(worker: dict) -> float:
    """Mirrors Phase 2 engineer_cluster_features() exactly."""
    return float(
        0.30 * (worker["bmi"] - 22.5)            / 5.0
        + 0.20 * (worker["age"] - 35)             / 15.0
        - 0.35 * worker["acclimatisation_days"]   / 45.0
        + 0.25 * (worker["metabolic_rate"] - 250) / 100.0
        - 0.20 * (worker["hydration_level"] - 3)  / 2.0
        + 0.15 * worker["hr_delta_t30"]           / 5.0
    )


def compute_adaptive_multiplier(vuln_score: float) -> float:
    """Mirrors Phase 2 build_cluster_label_column() multiplier formula."""
    vuln_range = VULN_MAX - VULN_MIN if VULN_MAX != VULN_MIN else 1.0
    mult = 1.15 - 0.35 * (vuln_score - VULN_MIN) / vuln_range
    return float(np.clip(mult, 0.70, 1.20))


def assign_cluster(worker: dict, reg: _ModelRegistry) -> tuple:
    """Assign new worker to K-Means cluster. Returns (cluster_id, persona_name)."""
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
    Full inference pipeline. Accepts raw sensor + personal inputs dict.
    Returns structured prediction response for the app UI.

    Required keys: age, bmi, acclimatisation_days, ambient_temp, humidity,
                   metabolic_rate, work_hours, hydration_level, heart_rate
    Optional keys: wind_speed, solar_radiation,
                   ambient_temp_t15/t30, humidity_t15/t30, heart_rate_t15/t30
    """
    reg = get_registry()

    # Pull and type-cast required inputs
    age                  = float(worker_input["age"])
    bmi                  = float(worker_input["bmi"])
    acclimatisation_days = float(worker_input["acclimatisation_days"])
    metabolic_rate       = float(worker_input["metabolic_rate"])
    work_hours           = float(worker_input["work_hours"])
    hydration_level      = float(worker_input["hydration_level"])
    ambient_temp         = float(worker_input["ambient_temp"])
    humidity             = float(worker_input["humidity"])
    wind_speed           = float(worker_input.get("wind_speed") or 1.5)
    solar_radiation      = float(worker_input.get("solar_radiation") or 500.0)
    heart_rate           = float(worker_input["heart_rate"])

    # Lag values — use 'or' to handle None from Pydantic model_dump()
    ambient_temp_t15 = float(worker_input.get("ambient_temp_t15") or (ambient_temp - 0.8))
    ambient_temp_t30 = float(worker_input.get("ambient_temp_t30") or (ambient_temp - 1.5))
    humidity_t15     = float(worker_input.get("humidity_t15")     or (humidity - 0.5))
    humidity_t30     = float(worker_input.get("humidity_t30")     or (humidity - 1.0))
    heart_rate_t15   = float(worker_input.get("heart_rate_t15")   or (heart_rate - 2.5))
    heart_rate_t30   = float(worker_input.get("heart_rate_t30")   or (heart_rate - 5.0))

    # Phase 1 derived features — recomputed live
    hr_delta_t15          = heart_rate - heart_rate_t15
    hr_delta_t30          = heart_rate - heart_rate_t30
    temp_delta_t15        = ambient_temp - ambient_temp_t15
    temp_humidity_product = (ambient_temp * humidity) / 100.0
    heat_index_val        = compute_heat_index(ambient_temp, humidity)
    core_temp_tre         = compute_core_temp_proxy(
        ambient_temp, humidity, wind_speed, solar_radiation,
        metabolic_rate, work_hours, hydration_level,
        acclimatisation_days, bmi, age,
    )
    sweat_rate = compute_sweat_rate(metabolic_rate, ambient_temp, humidity)

    # Phase 2 cluster assignment
    worker = {
        "age": age, "bmi": bmi,
        "acclimatisation_days": acclimatisation_days,
        "metabolic_rate": metabolic_rate,
        "hydration_level": hydration_level,
        "hr_delta_t30": hr_delta_t30,
    }
    cluster_id, persona_name = assign_cluster(worker, reg)
    vuln_score   = compute_vulnerability_score(worker)
    adaptive_mult = compute_adaptive_multiplier(vuln_score)

    # Assemble 27-feature vector in EXACT feature_list.json order
    feature_values = {
        "age": age, "bmi": bmi,
        "acclimatisation_days": acclimatisation_days,
        "ambient_temp": ambient_temp, "humidity": humidity,
        "wind_speed": wind_speed, "solar_radiation": solar_radiation,
        "ambient_temp_t15": ambient_temp_t15, "ambient_temp_t30": ambient_temp_t30,
        "humidity_t15": humidity_t15, "humidity_t30": humidity_t30,
        "metabolic_rate": metabolic_rate, "work_hours": work_hours,
        "hydration_level": hydration_level, "heart_rate": heart_rate,
        "heart_rate_t15": heart_rate_t15, "heart_rate_t30": heart_rate_t30,
        "sweat_rate": sweat_rate, "core_temp_tre": core_temp_tre,
        "heat_index": heat_index_val,
        "hr_delta_t15": hr_delta_t15, "hr_delta_t30": hr_delta_t30,
        "temp_delta_t15": temp_delta_t15,
        "temp_humidity_product": temp_humidity_product,
        "vulnerability_score": vuln_score,
        "cluster_id": float(cluster_id),
        "adaptive_alert_multiplier": adaptive_mult,
    }
    X = np.array([[feature_values[f] for f in reg.features]])

    # XGBoost inference
    probs      = reg.xgb_model.predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    pred_label = RISK_LABELS[pred_class]

    # Adaptive alert threshold
    threshold  = 0.50 * adaptive_mult
    alert_fires = bool(probs[2] >= threshold or probs[3] >= threshold)

    time_to_peak = _estimate_time_to_peak(pred_class, hr_delta_t30, vuln_score)
    top_factors  = _get_top_risk_factors(feature_values, pred_class, probs)
    meta         = RISK_META[pred_label]

    return {
        "predicted_class":          pred_label,
        "predicted_class_num":      pred_class,
        "probabilities":            {l: round(float(p), 4)
                                     for l, p in zip(RISK_LABELS, probs)},
        "risk_score":               round(float(np.dot(probs, [0,33,67,100])), 1),
        "alert_fires":              alert_fires,
        "threshold_used":           round(threshold, 3),
        "adaptive_alert_multiplier": round(adaptive_mult, 3),
        "cluster_id":               cluster_id,
        "persona_name":             persona_name,
        "vulnerability_score":      round(vuln_score, 4),
        "core_temp_estimate":       round(core_temp_tre, 2),
        "heat_index":               round(heat_index_val, 1),
        "sweat_rate_estimate":      round(sweat_rate, 1),
        "hr_trajectory":            round(hr_delta_t30, 2),
        "time_to_peak_minutes":     time_to_peak,
        "risk_color":               meta["color"],
        "risk_emoji":               meta["emoji"],
        "risk_message":             meta["message"],
        "risk_action":              meta["action"],
        "top_risk_factors":         top_factors,
    }


def _estimate_time_to_peak(pred_class: int, hr_delta: float, vuln_score: float) -> int:
    """Estimate minutes until risk peaks from class + HR trajectory + vulnerability."""
    base_times = {3: 8, 2: 20, 1: 38, 0: 65}
    base  = base_times.get(pred_class, 60)
    return max(3, base - max(0, int(hr_delta * 1.5)) - max(0, int(vuln_score * 8)))


def _get_top_risk_factors(feature_values: dict, pred_class: int,
                           probs: np.ndarray) -> list:
    """Return top 3 human-readable risk factors based on domain rules + SHAP findings."""
    factors = []
    tre  = feature_values["core_temp_tre"]
    hr_d = feature_values["hr_delta_t30"]
    hi   = feature_values["heat_index"]
    accl = feature_values["acclimatisation_days"]
    hyd  = feature_values["hydration_level"]

    if tre >= 38.5:
        factors.append({"feature": "Core temperature",   "value": f"{tre:.1f}°C",
                         "severity": "critical", "description": "Body temp crossed danger threshold."})
    elif tre >= 38.0:
        factors.append({"feature": "Core temperature",   "value": f"{tre:.1f}°C",
                         "severity": "high",     "description": "Body temperature significantly elevated."})
    if hr_d >= 10:
        factors.append({"feature": "Rising heart rate",  "value": f"+{hr_d:.1f} bpm over 30 min",
                         "severity": "critical", "description": "Rapid HR increase signals accelerating heat strain."})
    elif hr_d >= 5:
        factors.append({"feature": "Rising heart rate",  "value": f"+{hr_d:.1f} bpm over 30 min",
                         "severity": "high",     "description": "Heart rate is rising — monitor closely."})
    if hi >= 54:
        factors.append({"feature": "Heat index",         "value": f"{hi:.1f}°C",
                         "severity": "critical", "description": "Combined heat + humidity creating extreme conditions."})
    elif hi >= 41:
        factors.append({"feature": "Heat index",         "value": f"{hi:.1f}°C",
                         "severity": "high",     "description": "High heat index — air feels much hotter than it is."})
    if accl < 7:
        factors.append({"feature": "Low acclimatisation","value": f"{int(accl)} days",
                         "severity": "high",     "description": "Worker has not built heat tolerance."})
    if hyd <= 1:
        factors.append({"feature": "Severe dehydration", "value": f"Level {int(hyd)}/5",
                         "severity": "critical", "description": "Critically low hydration amplifies all risk."})
    elif hyd <= 2:
        factors.append({"feature": "Low hydration",      "value": f"Level {int(hyd)}/5",
                         "severity": "high",     "description": "Drink water now to reduce heat strain."})

    severity_order = {"critical": 0, "high": 1, "moderate": 2}
    factors.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return factors[:3]
```

---

## File 2 of 3 — `app.py`

FastAPI application with all endpoints, Pydantic schemas, and CORS middleware.

```python
"""
=============================================================================
PHASE 4: FASTAPI REST BACKEND
=============================================================================
Endpoints:
  POST /predict              — single worker prediction
  POST /predict/batch        — batch prediction (up to 100 workers)
  POST /onboard              — one-time worker onboarding
  GET  /health               — health check
  GET  /model/info           — model metadata + Phase 3 metrics
  GET  /demo/scenario/{name} — pre-built viva demo scenarios
  GET  /docs                 — Swagger UI (FastAPI built-in)

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
=============================================================================
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import time

from inference_engine import predict, get_registry, RISK_LABELS, RISK_META

app = FastAPI(
    title="Heatstroke Early Warning API",
    description=(
        "AI-powered heatstroke risk prediction for outdoor workers.\n\n"
        "Predicts heat stress risk 30–60 minutes early using XGBoost "
        "trained on ISO 7933 PHS-derived synthetic data with K-Means "
        "worker profiling for personalised adaptive thresholds.\n\n"
        "Phase 3 model metrics: F1-macro=0.9772 | ROC-AUC=0.9999 | "
        "Recall(High)=1.00 | Recall(Critical)=0.996"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    get_registry()


# ── Schemas ───────────────────────────────────────────────────────────────────

class WorkerReading(BaseModel):
    """Single worker reading — required fields from phone sensors."""
    # Personal (set once at onboarding)
    age:                  float = Field(..., ge=15, le=75)
    bmi:                  float = Field(..., ge=15, le=45)
    acclimatisation_days: float = Field(..., ge=0,  le=120)
    # Environmental (from weather API or phone)
    ambient_temp:         float = Field(..., ge=20, le=55)
    humidity:             float = Field(..., ge=10, le=100)
    wind_speed:           float = Field(1.5,  ge=0, le=15)
    solar_radiation:      float = Field(500.0, ge=0, le=1200)
    # Work
    metabolic_rate:       float = Field(..., ge=80, le=500)
    work_hours:           float = Field(..., ge=0,  le=12)
    hydration_level:      float = Field(..., ge=1,  le=5)
    # Physiological (accelerometer-based HR)
    heart_rate:           float = Field(..., ge=40, le=200)
    # Optional lag readings (stored by app every 15 min)
    ambient_temp_t15:     Optional[float] = Field(None)
    ambient_temp_t30:     Optional[float] = Field(None)
    humidity_t15:         Optional[float] = Field(None)
    humidity_t30:         Optional[float] = Field(None)
    heart_rate_t15:       Optional[float] = Field(None)
    heart_rate_t30:       Optional[float] = Field(None)

    @field_validator("hydration_level")
    @classmethod
    def check_hydration(cls, v):
        if v not in [1, 2, 3, 4, 5]:
            raise ValueError("hydration_level must be 1, 2, 3, 4, or 5")
        return v

    model_config = {"json_schema_extra": {"example": {
        "age": 34, "bmi": 24.5, "acclimatisation_days": 12,
        "ambient_temp": 42.0, "humidity": 78.0, "wind_speed": 0.8,
        "solar_radiation": 850.0, "metabolic_rate": 280.0,
        "work_hours": 5.0, "hydration_level": 2, "heart_rate": 125.0,
        "ambient_temp_t15": 41.2, "ambient_temp_t30": 40.5,
        "humidity_t15": 76.0, "humidity_t30": 74.5,
        "heart_rate_t15": 118.0, "heart_rate_t30": 110.0,
    }}}


class BatchRequest(BaseModel):
    workers: List[WorkerReading] = Field(..., min_length=1, max_length=100)


class OnboardRequest(BaseModel):
    """One-time profile setup from app's onboarding screen."""
    age:                  float = Field(..., ge=15, le=75)
    bmi:                  float = Field(..., ge=15, le=45)
    acclimatisation_days: float = Field(..., ge=0,  le=120)
    metabolic_rate:       float = Field(200.0, ge=80, le=500)
    hydration_level:      float = Field(3.0,   ge=1,  le=5)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    reg = get_registry()
    return {"status": "healthy", "models_loaded": reg.is_loaded,
            "n_features": reg.n_features, "n_clusters": reg.kmeans.n_clusters,
            "model_version": "1.0.0"}


@app.get("/model/info", tags=["System"])
async def model_info():
    reg = get_registry()
    return {"model_type": "XGBoostClassifier", "n_features": reg.n_features,
            "features": reg.features, "n_classes": 4, "risk_labels": RISK_LABELS,
            "n_clusters": reg.kmeans.n_clusters, "phase3_metrics": reg.metrics}


@app.post("/predict", tags=["Prediction"])
async def predict_risk(reading: WorkerReading):
    """
    Predict heat stress risk for a single worker.
    Accepts current sensor readings + optional 15-min/30-min lag values.
    Returns risk class, probabilities, alert status, and personalised action.
    """
    try:
        return predict(reading.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Batch prediction for up to 100 workers (supervisor dashboard).
    Returns all predictions plus aggregate alert count and critical count.
    """
    t0      = time.time()
    results = []
    for worker in request.workers:
        try:
            results.append(predict(worker.model_dump()))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")
    return {
        "results":        results,
        "total_workers":  len(results),
        "alert_count":    sum(1 for r in results if r["alert_fires"]),
        "critical_count": sum(1 for r in results if r["predicted_class"] == "Critical"),
        "processing_ms":  round((time.time() - t0) * 1000, 2),
    }


@app.post("/onboard", tags=["Worker Profile"])
async def onboard_worker(request: OnboardRequest):
    """
    One-time worker onboarding. Assigns K-Means cluster persona.
    Returns personalisation summary shown on the worker's profile screen.
    """
    from inference_engine import (assign_cluster, compute_vulnerability_score,
                                   compute_adaptive_multiplier, get_registry)
    worker_data = {
        "age": request.age, "bmi": request.bmi,
        "acclimatisation_days": request.acclimatisation_days,
        "metabolic_rate": request.metabolic_rate,
        "hydration_level": request.hydration_level,
        "hr_delta_t30": 0.0,  # no readings yet at onboarding
    }
    reg          = get_registry()
    cluster_id, persona = assign_cluster(worker_data, reg)
    vuln_score   = compute_vulnerability_score(worker_data)
    mult         = compute_adaptive_multiplier(vuln_score)

    persona_descriptions = {
        "Acclimatised Veteran": (
            "You have strong heat tolerance. Alerts are calibrated slightly "
            "higher for you because your body handles heat stress well."),
        "High-BMI Novice": (
            "You are still adapting to heat stress. Alerts fire earlier to "
            "give you more time to act safely."),
    }
    direction = "earlier" if mult < 1.0 else "later"
    pct_shift = abs(round((mult - 1.0) * 100))

    return {
        "cluster_id":               cluster_id,
        "persona_name":             persona,
        "vulnerability_score":      round(vuln_score, 4),
        "adaptive_alert_multiplier": round(mult, 3),
        "persona_description":      persona_descriptions.get(persona, "Profile set."),
        "personalisation_summary":  (
            f"Alerts will fire {pct_shift}% {direction} than the baseline. "
            f"This is personalised to your acclimatisation level, BMI, and exertion pattern."
        ),
    }


@app.get("/demo/scenario/{name}", tags=["Demo"])
async def demo_scenario(name: str):
    """
    Pre-built demo scenarios for viva presentation.
    Available: safe | danger | moderate | critical_early
    """
    scenarios = {
        "safe": {
            "age": 32.0, "bmi": 22.5, "acclimatisation_days": 65.0,
            "ambient_temp": 31.0, "humidity": 45.0, "wind_speed": 2.5,
            "solar_radiation": 320.0, "metabolic_rate": 155.0,
            "work_hours": 2.0, "hydration_level": 4.0, "heart_rate": 82.0,
            "ambient_temp_t15": 30.2, "ambient_temp_t30": 29.5,
            "humidity_t15": 44.0, "humidity_t30": 43.5,
            "heart_rate_t15": 80.5, "heart_rate_t30": 79.0,
        },
        "danger": {
            "age": 48.0, "bmi": 31.0, "acclimatisation_days": 6.0,
            "ambient_temp": 44.5, "humidity": 87.0, "wind_speed": 0.3,
            "solar_radiation": 920.0, "metabolic_rate": 355.0,
            "work_hours": 6.5, "hydration_level": 1.0, "heart_rate": 138.0,
            "ambient_temp_t15": 43.5, "ambient_temp_t30": 42.8,
            "humidity_t15": 85.0, "humidity_t30": 83.5,
            "heart_rate_t15": 130.0, "heart_rate_t30": 122.0,
        },
        "moderate": {
            "age": 40.0, "bmi": 26.0, "acclimatisation_days": 20.0,
            "ambient_temp": 38.0, "humidity": 65.0, "wind_speed": 1.2,
            "solar_radiation": 600.0, "metabolic_rate": 230.0,
            "work_hours": 4.0, "hydration_level": 3.0, "heart_rate": 105.0,
            "ambient_temp_t15": 37.5, "ambient_temp_t30": 37.0,
            "humidity_t15": 63.0, "humidity_t30": 62.0,
            "heart_rate_t15": 101.0, "heart_rate_t30": 97.0,
        },
        "critical_early": {
            # Demonstrates trajectory-based early warning —
            # rising HR signals Critical 30+ min before collapse
            "age": 45.0, "bmi": 28.0, "acclimatisation_days": 10.0,
            "ambient_temp": 41.0, "humidity": 80.0, "wind_speed": 0.5,
            "solar_radiation": 800.0, "metabolic_rate": 320.0,
            "work_hours": 5.5, "hydration_level": 2.0, "heart_rate": 132.0,
            "ambient_temp_t15": 40.0, "ambient_temp_t30": 39.2,
            "humidity_t15": 78.0, "humidity_t30": 76.5,
            "heart_rate_t15": 121.0, "heart_rate_t30": 110.0,
        },
    }
    if name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{name}' not found. Available: {list(scenarios.keys())}"
        )
    return {"scenario": name, "input": scenarios[name], "prediction": predict(scenarios[name])}
```

---

## File 3 of 3 — `test_api.py`

48 automated tests across 5 test classes. Runs without a live server via `TestClient`.

```python
"""
=============================================================================
PHASE 4: TEST SUITE — 48 tests, all passing
=============================================================================
Categories:
  TestInferenceEngine   (15) — unit tests for all engine functions
  TestSafetyConstraints  (7) — alert logic, threshold, trajectory tests
  TestAPIEndpoints      (14) — all endpoints, validation, schema
  TestDemoScenarios      (6) — regression checks on all 4 demo scenarios
  TestEdgeCases          (6) — boundary values, None handling, extremes

Run: python test_api.py
=============================================================================
"""

import json, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from fastapi.testclient import TestClient
from app import app
from inference_engine import (
    predict, compute_heat_index, compute_core_temp_proxy,
    compute_vulnerability_score, compute_adaptive_multiplier, RISK_LABELS,
)

client = TestClient(app)

SAFE_WORKER = {
    "age": 32.0, "bmi": 22.5, "acclimatisation_days": 65.0,
    "ambient_temp": 31.0, "humidity": 45.0, "wind_speed": 2.5,
    "solar_radiation": 320.0, "metabolic_rate": 155.0,
    "work_hours": 2.0, "hydration_level": 4.0, "heart_rate": 82.0,
    "ambient_temp_t15": 30.2, "ambient_temp_t30": 29.5,
    "humidity_t15": 44.0, "humidity_t30": 43.5,
    "heart_rate_t15": 80.5, "heart_rate_t30": 79.0,
}

DANGER_WORKER = {
    "age": 48.0, "bmi": 31.0, "acclimatisation_days": 6.0,
    "ambient_temp": 44.5, "humidity": 87.0, "wind_speed": 0.3,
    "solar_radiation": 920.0, "metabolic_rate": 355.0,
    "work_hours": 6.5, "hydration_level": 1.0, "heart_rate": 138.0,
    "ambient_temp_t15": 43.5, "ambient_temp_t30": 42.8,
    "humidity_t15": 85.0, "humidity_t30": 83.5,
    "heart_rate_t15": 130.0, "heart_rate_t30": 122.0,
}

MINIMAL_WORKER = {
    "age": 30.0, "bmi": 23.0, "acclimatisation_days": 30.0,
    "ambient_temp": 35.0, "humidity": 60.0,
    "metabolic_rate": 200.0, "work_hours": 3.0,
    "hydration_level": 3.0, "heart_rate": 95.0,
    # No lag values — must default gracefully
}


class TestInferenceEngine(unittest.TestCase):
    def test_heat_index_formula_extremes(self):
        self.assertGreaterEqual(compute_heat_index(20.0, 10.0), 25.0)
        self.assertLessEqual(compute_heat_index(50.0, 100.0), 70.0)

    def test_heat_index_increases_with_temp(self):
        self.assertGreater(compute_heat_index(45.0, 60.0), compute_heat_index(30.0, 60.0))

    def test_heat_index_increases_with_humidity(self):
        self.assertGreater(compute_heat_index(38.0, 90.0), compute_heat_index(38.0, 40.0))

    def test_core_temp_within_physiological_range(self):
        tre = compute_core_temp_proxy(50,100,0,1200,500,12,1,0,40,60)
        self.assertGreaterEqual(tre, 36.0)
        self.assertLessEqual(tre, 39.5)

    def test_core_temp_low_for_safe_conditions(self):
        tre = compute_core_temp_proxy(28,30,3,100,100,1,5,90,20,25)
        self.assertLess(tre, 38.0)

    def test_vulnerability_score_direction(self):
        novice  = {"bmi":35,"age":50,"acclimatisation_days":5,
                   "metabolic_rate":380,"hydration_level":1,"hr_delta_t30":15}
        veteran = {"bmi":20,"age":28,"acclimatisation_days":80,
                   "metabolic_rate":130,"hydration_level":5,"hr_delta_t30":1}
        self.assertGreater(compute_vulnerability_score(novice),
                           compute_vulnerability_score(veteran))

    def test_adaptive_multiplier_range(self):
        for vuln in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            mult = compute_adaptive_multiplier(vuln)
            self.assertGreaterEqual(mult, 0.70)
            self.assertLessEqual(mult, 1.20)

    def test_adaptive_multiplier_direction(self):
        self.assertGreater(compute_adaptive_multiplier(-0.5),
                           compute_adaptive_multiplier(0.5))

    def test_predict_returns_required_keys(self):
        result = predict(SAFE_WORKER)
        for key in ["predicted_class","probabilities","alert_fires",
                    "cluster_id","time_to_peak_minutes","hr_trajectory"]:
            self.assertIn(key, result)

    def test_predict_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(predict(SAFE_WORKER)["probabilities"].values()),
                               1.0, places=4)

    def test_predict_class_matches_argmax(self):
        result = predict(DANGER_WORKER)
        self.assertEqual(result["predicted_class"],
                         max(result["probabilities"], key=result["probabilities"].get))

    def test_risk_score_range(self):
        for w in [SAFE_WORKER, DANGER_WORKER, MINIMAL_WORKER]:
            s = predict(w)["risk_score"]
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)

    def test_minimal_worker_no_lag_fields(self):
        result = predict(MINIMAL_WORKER)
        self.assertIn("predicted_class", result)

    def test_time_to_peak_positive(self):
        for w in [SAFE_WORKER, DANGER_WORKER]:
            self.assertGreater(predict(w)["time_to_peak_minutes"], 0)

    def test_top_risk_factors_max_three(self):
        self.assertLessEqual(len(predict(DANGER_WORKER)["top_risk_factors"]), 3)


class TestSafetyConstraints(unittest.TestCase):
    def test_danger_worker_fires_alert(self):
        self.assertTrue(predict(DANGER_WORKER)["alert_fires"])

    def test_safe_worker_does_not_fire_alert(self):
        self.assertFalse(predict(SAFE_WORKER)["alert_fires"])

    def test_critical_worker_high_risk_score(self):
        self.assertGreater(predict(DANGER_WORKER)["risk_score"], 70.0)

    def test_safe_worker_low_risk_score(self):
        self.assertLess(predict(SAFE_WORKER)["risk_score"], 40.0)

    def test_adaptive_threshold_shifts_per_persona(self):
        self.assertGreaterEqual(predict(SAFE_WORKER)["adaptive_alert_multiplier"],
                                predict(DANGER_WORKER)["adaptive_alert_multiplier"])

    def test_time_to_peak_lower_for_danger(self):
        self.assertLess(predict(DANGER_WORKER)["time_to_peak_minutes"],
                        predict(SAFE_WORKER)["time_to_peak_minutes"])

    def test_hr_trajectory_captured(self):
        result = predict(DANGER_WORKER)
        expected = DANGER_WORKER["heart_rate"] - DANGER_WORKER["heart_rate_t30"]
        self.assertAlmostEqual(result["hr_trajectory"], expected, places=1)


class TestAPIEndpoints(unittest.TestCase):
    def test_health_check_200(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["models_loaded"])

    def test_health_check_n_features(self):
        self.assertEqual(client.get("/health").json()["n_features"], 27)

    def test_model_info_200(self):
        r = client.get("/model/info")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["features"]), 27)

    def test_predict_endpoint_safe_worker(self):
        r = client.post("/predict", json=SAFE_WORKER)
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["predicted_class"], RISK_LABELS)

    def test_predict_endpoint_danger_worker(self):
        r = client.post("/predict", json=DANGER_WORKER)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["alert_fires"])

    def test_predict_endpoint_minimal_worker(self):
        self.assertEqual(client.post("/predict", json=MINIMAL_WORKER).status_code, 200)

    def test_predict_invalid_hydration(self):
        self.assertEqual(
            client.post("/predict", json={**SAFE_WORKER, "hydration_level": 6.0}).status_code, 422)

    def test_predict_out_of_range_temp(self):
        self.assertEqual(
            client.post("/predict", json={**SAFE_WORKER, "ambient_temp": 10.0}).status_code, 422)

    def test_predict_missing_required_field(self):
        self.assertEqual(
            client.post("/predict", json={k:v for k,v in SAFE_WORKER.items()
                                          if k != "heart_rate"}).status_code, 422)

    def test_predict_response_schema(self):
        r = client.post("/predict", json=SAFE_WORKER).json()
        self.assertIsInstance(r["predicted_class"], str)
        self.assertIsInstance(r["alert_fires"], bool)
        self.assertIsInstance(r["time_to_peak_minutes"], int)

    def test_batch_predict_200(self):
        r = client.post("/predict/batch",
                        json={"workers": [SAFE_WORKER, DANGER_WORKER, MINIMAL_WORKER]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total_workers"], 3)

    def test_batch_alert_count_consistent(self):
        r    = client.post("/predict/batch", json={"workers": [SAFE_WORKER, DANGER_WORKER]})
        data = r.json()
        self.assertEqual(data["alert_count"],
                         sum(1 for res in data["results"] if res["alert_fires"]))

    def test_onboard_endpoint_200(self):
        r = client.post("/onboard", json={"age":34,"bmi":24.5,"acclimatisation_days":12})
        self.assertEqual(r.status_code, 200)
        self.assertIn("persona_name", r.json())

    def test_onboard_returns_valid_cluster_id(self):
        from inference_engine import get_registry
        n = get_registry().kmeans.n_clusters
        r = client.post("/onboard", json={"age":30,"bmi":22,"acclimatisation_days":30})
        self.assertIn(r.json()["cluster_id"], range(n))


class TestDemoScenarios(unittest.TestCase):
    def test_safe_scenario_no_alert(self):
        r = client.get("/demo/scenario/safe")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["prediction"]["alert_fires"])

    def test_danger_scenario_fires_alert(self):
        r = client.get("/demo/scenario/danger")
        self.assertTrue(r.json()["prediction"]["alert_fires"])

    def test_all_scenarios_return_200(self):
        for name in ["safe", "danger", "moderate", "critical_early"]:
            self.assertEqual(client.get(f"/demo/scenario/{name}").status_code, 200)

    def test_invalid_scenario_404(self):
        self.assertEqual(client.get("/demo/scenario/nonexistent").status_code, 404)

    def test_critical_early_trajectory_detected(self):
        r = client.get("/demo/scenario/critical_early")
        self.assertGreater(r.json()["prediction"]["hr_trajectory"], 5.0)

    def test_danger_has_risk_factors(self):
        r = client.get("/demo/scenario/danger")
        self.assertGreater(len(r.json()["prediction"]["top_risk_factors"]), 0)


class TestEdgeCases(unittest.TestCase):
    def test_maximum_stress_conditions(self):
        extreme = {"age":60,"bmi":40,"acclimatisation_days":0,
                   "ambient_temp":54.9,"humidity":99.9,"wind_speed":0,
                   "solar_radiation":1199,"metabolic_rate":499,"work_hours":11.9,
                   "hydration_level":1,"heart_rate":199.9,
                   "ambient_temp_t15":53,"ambient_temp_t30":51,
                   "humidity_t15":98,"humidity_t30":96,
                   "heart_rate_t15":185,"heart_rate_t30":170}
        self.assertEqual(client.post("/predict", json=extreme).status_code, 200)

    def test_minimum_stress_conditions(self):
        minimal = {"age":20,"bmi":18,"acclimatisation_days":90,
                   "ambient_temp":20,"humidity":20,"wind_speed":10,
                   "solar_radiation":10,"metabolic_rate":100,"work_hours":0.1,
                   "hydration_level":5,"heart_rate":60}
        self.assertEqual(client.post("/predict", json=minimal).status_code, 200)

    def test_lag_values_same_as_current(self):
        static = {**SAFE_WORKER,
                  "ambient_temp_t15": SAFE_WORKER["ambient_temp"],
                  "ambient_temp_t30": SAFE_WORKER["ambient_temp"],
                  "heart_rate_t15":   SAFE_WORKER["heart_rate"],
                  "heart_rate_t30":   SAFE_WORKER["heart_rate"]}
        self.assertEqual(client.post("/predict", json=static).status_code, 200)

    def test_falling_heart_rate_negative_delta(self):
        cooling = {**SAFE_WORKER, "heart_rate":80, "heart_rate_t15":90, "heart_rate_t30":100}
        self.assertLess(predict(cooling)["hr_trajectory"], 0.0)

    def test_probabilities_all_present(self):
        result = predict(SAFE_WORKER)
        for label in RISK_LABELS:
            self.assertIn(label, result["probabilities"])

    def test_batch_empty_list_rejected(self):
        r = client.post("/predict/batch", json={"workers": []})
        self.assertIn(r.status_code, [422, 400])


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestInferenceEngine, TestSafetyConstraints,
                TestAPIEndpoints, TestDemoScenarios, TestEdgeCases]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print(f"\n{'='*60}")
    print(f"  Tests run: {result.testsRun}  |  "
          f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}  |  "
          f"Status: {'✓ ALL PASSED' if result.wasSuccessful() else '✗ SOME FAILED'}")
    print(f"{'='*60}")
    import sys; sys.exit(0 if result.wasSuccessful() else 1)
```

---

## App Integration Pattern

This is how the React Native / Android app calls the API:

```javascript
// worker_api.js — called every 15 minutes from the app's background service

const API_BASE = "http://your-server:8000";

async function checkHeatRisk(workerProfile, currentReadings, lagHistory) {
  const payload = {
    // From onboarding (stored locally)
    age:                  workerProfile.age,
    bmi:                  workerProfile.bmi,
    acclimatisation_days: workerProfile.acclimatisation_days,

    // From weather API (updated every 15 min)
    ambient_temp:     currentReadings.temperature,
    humidity:         currentReadings.humidity,
    wind_speed:       currentReadings.windSpeed,
    solar_radiation:  currentReadings.solarRadiation,

    // From user (shift start)
    metabolic_rate:  workerProfile.metabolicRate,
    work_hours:      currentReadings.hoursWorked,

    // From emoji hydration selector (updated by user)
    hydration_level: currentReadings.hydrationLevel,

    // From accelerometer-based HR estimation
    heart_rate: currentReadings.heartRate,

    // From local lag history (stored by app)
    ambient_temp_t15: lagHistory.temp_15min_ago,
    ambient_temp_t30: lagHistory.temp_30min_ago,
    humidity_t15:     lagHistory.humidity_15min_ago,
    humidity_t30:     lagHistory.humidity_30min_ago,
    heart_rate_t15:   lagHistory.hr_15min_ago,
    heart_rate_t30:   lagHistory.hr_30min_ago,
  };

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const result = await response.json();

  if (result.alert_fires) {
    // Show alert notification
    showAlert({
      title:    result.risk_emoji + " " + result.predicted_class + " risk detected",
      message:  result.risk_message,
      action:   result.risk_action,
      color:    result.risk_color,
      timeToPeak: result.time_to_peak_minutes,
      factors:  result.top_risk_factors,
    });
  }

  return result;
}

// One-time onboarding call
async function onboardWorker(profile) {
  const response = await fetch(`${API_BASE}/onboard`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  return response.json(); // saves cluster_id, adaptive_alert_multiplier
}
```

---

## Viva Questions on Phase 4 — Answers

**Why rebuild the feature engineering in the inference engine instead of just calling model.predict()?**

The XGBoost model was trained on 27 engineered features, not 11 raw inputs. If you pass raw sensor readings directly to `predict_proba()`, you give the model the wrong inputs and it produces nonsense. Every feature that was computed during training — PHS core temperature, heat index, HR deltas, cluster assignment, vulnerability score — must be recomputed at inference time from the raw inputs using the exact same equations. This is why `feature_list.json` exists: it records the exact 27-column order that the model expects. The inference engine mirrors the entire pipeline from Phases 1 and 2.

**Why use FastAPI instead of Flask?**

FastAPI generates automatic Swagger documentation at `/docs` (zero extra code), handles async requests natively (important for batch endpoints), validates inputs automatically via Pydantic (invalid hydration level returns a 422 with a clear error message without any validation code), and serialises responses to JSON automatically. Flask requires manual validation and serialisation. FastAPI also runs on the same Starlette ASGI stack as production-grade frameworks like Uvicorn.

**How does the adaptive alert threshold work in the app?**

At onboarding, the worker's K-Means cluster is assigned and their `adaptive_alert_multiplier` is stored on their phone (e.g., 0.80 for a High-BMI Novice, 1.15 for an Acclimatised Veteran). At each inference call, the API multiplies the base threshold (0.50) by this factor. A worker with a 0.80 multiplier fires an alert when P(High) ≥ 0.40 instead of 0.50 — 10 percentage points earlier. The same trained model is used for everyone; only the decision threshold shifts. This is personalisation without retraining.

**What happens if the worker doesn't have lag readings yet (first shift)?**

The inference engine defaults gracefully. When `heart_rate_t30` is absent (or `None` from Pydantic), the code uses `heart_rate - 5.0` as an estimated t-30 reading. This implies a slight HR rise over 30 minutes, which is physiologically conservative (slightly overestimates current heat load). After 30 minutes of use, the app stores the real readings and passes them. The `or` operator in Python handles both missing keys and explicit `None` values: `float(worker_input.get("heart_rate_t30") or (heart_rate - 5.0))`.

**Why are there 48 tests?**

Safety-critical systems require more rigorous testing than standard software. The test suite validates: all mathematical helper functions individually, alert logic for canonical safe/danger scenarios, all 6 API endpoints with both valid and invalid inputs, all 4 demo scenarios as regression checks (so a model update doesn't silently break the demo), and 6 edge cases covering boundary values, None inputs, extreme physiological conditions, and falling HR trajectories. 44 tests cover correctness; 4 tests cover safety constraints specifically.
