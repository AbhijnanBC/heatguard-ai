"""
=============================================================================
PHASE 4: FASTAPI REST BACKEND
=============================================================================
Exposes the heatstroke prediction model as a REST API.

Endpoints:
    POST /predict             — main prediction (single worker)
    POST /predict/batch       — batch prediction (up to 100 workers)
    POST /onboard             — one-time worker onboarding (assigns cluster)
    GET  /health              — health check
    GET  /model/info          — model metadata and metrics
    GET  /demo/scenario/{name}— run a pre-built demo scenario
    GET  /docs                — auto-generated Swagger UI (FastAPI built-in)

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    Test from Python:
        import requests
        r = requests.post("http://localhost:8000/predict", json={...})
        print(r.json())
=============================================================================
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import time

from inference_engine import predict, get_registry, RISK_LABELS, RISK_META

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

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

# Allow all origins for development (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload models at startup
@app.on_event("startup")
async def startup_event():
    get_registry()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class WorkerReading(BaseModel):
    """
    Single worker reading. Required fields = what a smartphone can provide.
    Optional lag fields = collected every 15 minutes; app stores them locally.
    """
    # Personal (from onboarding — set once)
    age:                  float = Field(..., ge=15, le=75,  description="Worker age in years")
    bmi:                  float = Field(..., ge=15, le=45,  description="Body Mass Index kg/m²")
    acclimatisation_days: float = Field(..., ge=0,  le=120, description="Days of heat acclimatisation")

    # Environmental (from weather API or phone sensors)
    ambient_temp:         float = Field(..., ge=20, le=55,  description="Ambient temperature °C")
    humidity:             float = Field(..., ge=10, le=100, description="Relative humidity %")
    wind_speed:           float = Field(1.5,  ge=0,  le=15,  description="Wind speed m/s")
    solar_radiation:      float = Field(500.0, ge=0, le=1200, description="Solar radiation W/m²")

    # Work parameters
    metabolic_rate:       float = Field(..., ge=80, le=500, description="Metabolic work rate W (ISO 8996)")
    work_hours:           float = Field(..., ge=0,  le=12,  description="Hours elapsed in current shift")
    hydration_level:      float = Field(..., ge=1,  le=5,   description="Hydration level 1=very low 5=full")

    # Physiological (from accelerometer-based HR estimation)
    heart_rate:           float = Field(..., ge=40, le=200, description="Current heart rate bpm")

    # Lag readings (stored by app from 15-min and 30-min ago)
    ambient_temp_t15:     Optional[float] = Field(None, description="Ambient temp 15 min ago °C")
    ambient_temp_t30:     Optional[float] = Field(None, description="Ambient temp 30 min ago °C")
    humidity_t15:         Optional[float] = Field(None, description="Humidity 15 min ago %")
    humidity_t30:         Optional[float] = Field(None, description="Humidity 30 min ago %")
    heart_rate_t15:       Optional[float] = Field(None, description="Heart rate 15 min ago bpm")
    heart_rate_t30:       Optional[float] = Field(None, description="Heart rate 30 min ago bpm")

    @field_validator("hydration_level")
    @classmethod
    def check_hydration(cls, v):
        if v not in [1, 2, 3, 4, 5]:
            raise ValueError("hydration_level must be 1, 2, 3, 4, or 5")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "age": 34, "bmi": 24.5, "acclimatisation_days": 12,
            "ambient_temp": 42.0, "humidity": 78.0, "wind_speed": 0.8,
            "solar_radiation": 850.0, "metabolic_rate": 280.0,
            "work_hours": 5.0, "hydration_level": 2,
            "heart_rate": 125.0,
            "ambient_temp_t15": 41.2, "ambient_temp_t30": 40.5,
            "humidity_t15": 76.0, "humidity_t30": 74.5,
            "heart_rate_t15": 118.0, "heart_rate_t30": 110.0,
        }
    }}


class PredictionResponse(BaseModel):
    # Primary output
    predicted_class:          str
    predicted_class_num:      int
    probabilities:            dict
    risk_score:               float   # 0–100

    # Alert
    alert_fires:              bool
    threshold_used:           float
    adaptive_alert_multiplier: float

    # Worker profile
    cluster_id:               int
    persona_name:             str
    vulnerability_score:      float

    # Physiological estimates
    core_temp_estimate:       float
    heat_index:               float
    sweat_rate_estimate:      float
    hr_trajectory:            float

    # Time prediction
    time_to_peak_minutes:     int

    # UI metadata
    risk_color:               str
    risk_emoji:               str
    risk_message:             str
    risk_action:              str

    # Explainability
    top_risk_factors:         list


class BatchRequest(BaseModel):
    workers: List[WorkerReading] = Field(..., min_length=1, max_length=100,
                                         description="Up to 100 workers")


class BatchResponse(BaseModel):
    results:         List[PredictionResponse]
    total_workers:   int
    alert_count:     int
    critical_count:  int
    processing_ms:   float


class OnboardRequest(BaseModel):
    """One-time profile setup from the app's onboarding screen."""
    age:                  float = Field(..., ge=15, le=75)
    bmi:                  float = Field(..., ge=15, le=45)
    acclimatisation_days: float = Field(..., ge=0,  le=120)
    metabolic_rate:       float = Field(200.0, ge=80, le=500,
                                        description="Estimated metabolic rate for their job type")
    hydration_level:      float = Field(3.0,   ge=1,  le=5)

    model_config = {"json_schema_extra": {
        "example": {"age": 34, "bmi": 24.5, "acclimatisation_days": 12,
                    "metabolic_rate": 250.0, "hydration_level": 3}
    }}


class OnboardResponse(BaseModel):
    cluster_id:               int
    persona_name:             str
    vulnerability_score:      float
    adaptive_alert_multiplier: float
    persona_description:      str
    personalisation_summary:  str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# Note: I/O or trivial calls can remain async
@app.get("/health", tags=["System"])
async def health_check():
    """Health check. Returns 200 if models are loaded and ready."""
    reg = get_registry()
    return {
        "status":         "healthy",
        "models_loaded":  reg.is_loaded,
        "n_features":     reg.n_features,
        "n_clusters":     reg.kmeans.n_clusters,
        "model_version":  "1.0.0",
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model metadata and test-set evaluation metrics from Phase 3."""
    reg = get_registry()
    return {
        "model_type":     "XGBoostClassifier",
        "n_features":     reg.n_features,
        "features":       reg.features,
        "n_classes":      4,
        "risk_labels":    RISK_LABELS,
        "n_clusters":     reg.kmeans.n_clusters,
        "phase3_metrics": reg.metrics,
        "training_notes": (
            "Trained on 5,450 ISO 7933 PHS-derived synthetic worker records. "
            "SMOTE oversampling applied to training set only. "
            "Optuna Bayesian hyperparameter optimisation (30 trials). "
            "Evaluation metrics are from held-out 15% test set (never seen during training)."
        ),
    }

# [FIX APPLIED]: Removed `async` so FastAPI runs this blocking CPU-bound task in a separate thread.
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_risk(reading: WorkerReading):
    """
    Predict heat stress risk for a single worker.

    Accepts current sensor readings + optional 15-min and 30-min lag values.
    Returns risk class, probability distribution, alert status, and
    personalised recommendations based on the worker's K-Means cluster.
    """
    try:
        result = predict(reading.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

# [FIX APPLIED]: Removed `async`
@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Batch prediction for up to 100 workers (supervisor dashboard use case).
    Returns all predictions plus aggregate alert statistics.
    """
    t0      = time.time()
    results = []
    for worker in request.workers:
        try:
            results.append(predict(worker.model_dump()))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction error: {str(e)}"
            )

    alert_count    = sum(1 for r in results if r["alert_fires"])
    critical_count = sum(1 for r in results if r["predicted_class"] == "Critical")
    elapsed_ms     = round((time.time() - t0) * 1000, 2)

    return {
        "results":        results,
        "total_workers":  len(results),
        "alert_count":    alert_count,
        "critical_count": critical_count,
        "processing_ms":  elapsed_ms,
    }

# [FIX APPLIED]: Removed `async`
@app.post("/onboard", response_model=OnboardResponse, tags=["Worker Profile"])
def onboard_worker(request: OnboardRequest):
    """
    One-time worker onboarding (runs once when worker first installs the app).
    Assigns K-Means cluster persona and returns personalisation summary
    shown on the worker's profile screen.
    """
    from inference_engine import (
        assign_cluster, compute_vulnerability_score,
        compute_adaptive_multiplier, PERSONA_MAP
    )

    worker_data = {
        "age":                  request.age,
        "bmi":                  request.bmi,
        "acclimatisation_days": request.acclimatisation_days,
        "metabolic_rate":       request.metabolic_rate,
        "hydration_level":      request.hydration_level,
        "hr_delta_t30":         0.0,   # no readings yet at onboarding
    }

    reg         = get_registry()
    cluster_id, persona = assign_cluster(worker_data, reg)
    vuln_score  = compute_vulnerability_score(worker_data)
    mult        = compute_adaptive_multiplier(vuln_score)

    persona_descriptions = {
        "Acclimatised Veteran": (
            "You have strong heat tolerance built over many days of outdoor work. "
            "Your body adapts well to heat stress. "
            "Alerts are calibrated slightly higher for you."
        ),
        "High-BMI Novice": (
            "You are earlier in your acclimatisation journey. "
            "Your body is still adapting to heat stress. "
            "Alerts are calibrated earlier to give you more time to act."
        ),
        "Young High-Exertion": (
            "You do high-intensity outdoor work. Your cardiovascular system is "
            "efficient but faces high demands. Alerts focus on your exertion rate."
        ),
    }

    multiplier_direction = "earlier" if mult < 1.0 else "later"
    pct_shift = abs(round((mult - 1.0) * 100))

    return {
        "cluster_id":               cluster_id,
        "persona_name":             persona,
        "vulnerability_score":      round(vuln_score, 4),
        "adaptive_alert_multiplier": round(mult, 3),
        "persona_description":      persona_descriptions.get(
            persona,
            "Your risk profile has been set up. Alerts are personalised for you."
        ),
        "personalisation_summary": (
            f"Based on your profile, alerts will fire {pct_shift}% {multiplier_direction} "
            f"than the baseline threshold. This accounts for your acclimatisation level, "
            f"BMI, and typical exertion pattern."
        ),
    }

# [FIX APPLIED]: Removed `async`
@app.get("/demo/scenario/{name}", tags=["Demo"])
def demo_scenario(name: str):
    """
    Run a pre-built demo scenario. Used for the viva live demo.

    Available scenarios:
        safe     — Acclimatised Veteran, mild conditions → Low risk, no alert
        danger   — High-BMI Novice, extreme heat, rising HR → Critical, alert fires
        moderate — Mid-range worker, borderline conditions
        critical_early — Shows how HR trajectory triggers alert 45 min early
    """
    scenarios = {
        "safe": {
            "age": 32.0, "bmi": 22.5, "acclimatisation_days": 65.0,
            "ambient_temp": 31.0, "humidity": 45.0, "wind_speed": 2.5,
            "solar_radiation": 320.0, "metabolic_rate": 155.0,
            "work_hours": 2.0, "hydration_level": 4.0,
            "heart_rate": 82.0,
            "ambient_temp_t15": 30.2, "ambient_temp_t30": 29.5,
            "humidity_t15": 44.0, "humidity_t30": 43.5,
            "heart_rate_t15": 80.5, "heart_rate_t30": 79.0,
        },
        "danger": {
            "age": 48.0, "bmi": 31.0, "acclimatisation_days": 6.0,
            "ambient_temp": 44.5, "humidity": 87.0, "wind_speed": 0.3,
            "solar_radiation": 920.0, "metabolic_rate": 355.0,
            "work_hours": 6.5, "hydration_level": 1.0,
            "heart_rate": 138.0,
            "ambient_temp_t15": 43.5, "ambient_temp_t30": 42.8,
            "humidity_t15": 85.0, "humidity_t30": 83.5,
            "heart_rate_t15": 130.0, "heart_rate_t30": 122.0,
        },
        "moderate": {
            "age": 40.0, "bmi": 26.0, "acclimatisation_days": 20.0,
            "ambient_temp": 38.0, "humidity": 65.0, "wind_speed": 1.2,
            "solar_radiation": 600.0, "metabolic_rate": 230.0,
            "work_hours": 4.0, "hydration_level": 3.0,
            "heart_rate": 105.0,
            "ambient_temp_t15": 37.5, "ambient_temp_t30": 37.0,
            "humidity_t15": 63.0, "humidity_t30": 62.0,
            "heart_rate_t15": 101.0, "heart_rate_t30": 97.0,
        },
        "critical_early": {
            # Demonstrates trajectory-based early warning:
            # Current readings look High, but rising HR signals Critical in 30 min
            "age": 45.0, "bmi": 28.0, "acclimatisation_days": 10.0,
            "ambient_temp": 41.0, "humidity": 80.0, "wind_speed": 0.5,
            "solar_radiation": 800.0, "metabolic_rate": 320.0,
            "work_hours": 5.5, "hydration_level": 2.0,
            "heart_rate": 132.0,
            "ambient_temp_t15": 40.0, "ambient_temp_t30": 39.2,
            "humidity_t15": 78.0, "humidity_t30": 76.5,
            "heart_rate_t15": 121.0, "heart_rate_t30": 110.0,
        },
    }

    if name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{name}' not found. "
                   f"Available: {list(scenarios.keys())}"
        )

    result = predict(scenarios[name])
    return {
        "scenario":    name,
        "input":       scenarios[name],
        "prediction":  result,
    }