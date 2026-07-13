<div align="center">

# 🔥 HeatGuard
### AI-Based Heatstroke Early Warning System for Outdoor Workers

**Predicting heat stress risk 30 minutes before it happens — personalised, explainable, and free.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-48%2F48%20passing-brightgreen.svg)](#-testing)
[![Deployment](https://img.shields.io/badge/deployment%20checks-14%2F14%20passed-brightgreen.svg)](#-deployment)

[Overview](#-overview) • [Architecture](#-architecture) • [Results](#-results) • [Quickstart](#-quickstart) • [Project Structure](#-project-structure) • [Demo](#-live-demo) • [Roadmap](#-roadmap)

</div>

---

## 🌍 Overview

Occupational heat stress is one of the most under-addressed public health crises of the climate era. India alone recorded **~48,000 heatstroke cases in 2024**, and the ILO estimates **167 billion working hours** are lost globally each year to rising temperatures — almost all of it among low-income outdoor workers who cannot afford wearable biosensors and are ignored by generic, one-size-fits-all heat advisories.

**HeatGuard** closes that gap. It's a smartphone-native, edge-deployable AI system that:

- 🔮 **Predicts 30–60 minutes ahead** — not current state, but where a worker's body is heading
- 👤 **Personalises every alert** — a 20-year veteran and a first-week novice get different thresholds, automatically
- ⚡ **Runs in ~1ms** — light enough for any mid-range Android phone, online or offline
- 🧠 **Explains itself** — every alert comes with SHAP-backed, plain-language reasoning
- 💸 **Costs nothing per worker** — no wearables, no subscriptions, just a phone and a weather API

Built entirely on a physiologically-grounded synthetic data pipeline (ISO 7933 PHS model), the system is validated end-to-end: data generation → worker profiling → model training → API → deployable app.

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   PHASE 1        │    │    PHASE 2       │    │     PHASE 3        │
│  Synthetic Data  │───▶│  K-Means Worker  │───▶│  XGBoost Training  │
│  (ISO 7933 PHS)  │    │    Profiling     │    │   + SHAP + Tuning  │
└─────────────────┘    └──────────────────┘    └───────────────────┘
   5,647 workers          4 vulnerability          F1=0.956
   27 features             personas               AUC=0.998
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                            PHASE 4                                │
│           FastAPI Inference Engine + 48 Automated Tests           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                            PHASE 5                                │
│     Worker App Prototype (HTML) · ONNX Export · Deployment Kit    │
└─────────────────────────────────────────────────────────────────┘
```

Each phase is fully self-contained with its own runner script, and the outputs of one phase feed directly into the next — a worker's `cluster_id` from Phase 2 becomes a live feature in the Phase 3 model, which is served by the Phase 4 API, which powers the Phase 5 app.

---

## 📊 Results

### Model Performance (held-out test set, never seen during training)

| Metric | Score | Why it matters |
|---|---|---|
| **F1-macro** | **0.9561** | Balanced performance across all 4 risk classes |
| **ROC-AUC (macro)** | **0.9984** | Near-perfect separation between risk levels |
| **Recall — Critical class** | **0.9857** | Catches ~99% of workers on the verge of collapse |
| **Recall — High class** | **0.9200** | Catches 92% of high-risk workers |
| Accuracy | 0.9750 | Reported, but *not* the headline metric (imbalanced data) |

> Sensitivity for High/Critical was treated as a **hard constraint**, not just a metric — hyperparameter trials that fell below 0.75/0.80 recall were automatically rejected during Optuna tuning, regardless of their F1 score. A missed alert is a categorically worse failure than an unnecessary one.

### What drives the predictions? (SHAP global importance)

| Rank | Feature | Role |
|---|---|---|
| 1 | `core_temp_tre` | ISO 7933 estimated core body temperature — the dominant signal |
| 2 | `acclimatisation_days` | Protective factor — heat adaptation |
| 3 | `cluster_id` | **Worker persona from Phase 2** — outranks raw temperature/humidity |
| 4 | `vulnerability_score` | Composite personalisation index |
| 5 | `heat_index` | Non-linear temp × humidity interaction |

The clustering stage isn't cosmetic — `cluster_id` ranks **above individual environmental sensors** in the model's decision-making, confirming that worker profiling genuinely improves prediction quality.

### System Validation

| Check | Result |
|---|---|
| Unit + integration + safety tests | **48 / 48 passing** |
| Deployment readiness checklist | **14 / 14 passed** |
| Inference latency | **~1.0 ms / prediction** |
| Exported model size (ONNX/JSON) | **< 250 KB** |

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/AbhijnanBC/heatstroke_ai.git
cd heatstroke_ai
pip install -r requirements.txt

# 2. Run the full pipeline, phase by phase
python src/phase1_generation/run_phase1.py
python src/phase2_clustering/run_phase2.py
python src/phase3_modeling/run_phase3.py

# 3. Launch the API
cd app/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# → Swagger docs at http://localhost:8000/docs

# 4. Run the test suite
cd ../tests
python test_api.py

# 5. Open the interactive app prototype (no server needed)
open ../../outputs/phase5/worker_app.html
```

### Try a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 48, "bmi": 31, "acclimatisation_days": 6,
    "ambient_temp": 44.5, "humidity": 87,
    "metabolic_rate": 355, "work_hours": 6.5,
    "hydration_level": 1, "heart_rate": 138,
    "heart_rate_t15": 130, "heart_rate_t30": 122
  }'
```

```json
{
  "predicted_class": "Critical",
  "risk_score": 100.0,
  "alert_fires": true,
  "persona_name": "High-BMI Novice",
  "time_to_peak_minutes": 3,
  "risk_action": "STOP WORK IMMEDIATELY. Seek shade and cool water. Call supervisor."
}
```

---

## 📁 Project Structure

```
heatstroke_ai/
├── app/
│   ├── api/
│   │   ├── app.py                  # FastAPI endpoints & schemas
│   │   └── inference_engine.py     # Feature engineering + prediction logic
│   └── tests/
│       └── test_api.py             # 48 automated tests
│
├── src/
│   ├── phase1_generation/          # ISO 7933 PHS synthetic data pipeline
│   ├── phase2_clustering/
│   │   ├── kmeans_profiling.py     # Clustering engine + vulnerability scoring
│   │   ├── run_phase2.py           # Entry point
│   │   └── visualise_phase2.py     # 7 diagnostic plots
│   └── phase3_modeling/
│       ├── train_xgboost.py        # Training, SMOTE, Optuna tuning, evaluation
│       ├── shap_interpreter.py     # SHAP explainability
│       ├── visualise_phase3.py     # 5 evaluation plots
│       └── run_phase3.py           # Entry point
│
├── data/                           # Generated datasets (CSV) at each phase
├── models/                         # Trained artefacts (.pkl, .json, feature list)
├── docs/                           # Model card, phase reports
│
└── outputs/
    ├── phase1_plots/               # Risk distribution, PHS validation, correlations
    ├── phase2_plots/               # Elbow curve, cluster radar, risk heatmap
    ├── phase3_plots/               # Confusion matrix, ROC, SHAP, demo scenarios
    └── phase5/
        ├── deploy/                 # Self-contained deployment package
        ├── export_and_deploy.py    # ONNX export + deployment checklist
        └── worker_app.html         # Interactive app prototype (open in browser)
```

---

## 🔬 How It Works

<details>
<summary><b>Phase 1 — Synthetic Data Generation</b></summary><br>

Generates 5,647 physiologically authentic worker records from the **ISO 7933 Predicted Heat Strain (PHS)** model rather than arbitrary noise. Key design choices:
- **Predictive labelling (t+30):** risk labels reflect a worker's projected state 30 minutes ahead, not their current reading
- **Non-linear synergy term:** temperature × humidity interaction scales exponentially past 35°C / 60% RH, mimicking real thermoregulatory failure
- **Dynamic HR lag physiology:** heart rate trajectory is driven by work hours, acclimatisation, and metabolic rate — not a static offset
- Gaussian sensor noise calibrated to real smartphone hardware accuracy
</details>

<details>
<summary><b>Phase 2 — K-Means Worker Profiling</b></summary><br>

Clusters workers into 4 personas using only **intrinsic physiological features** (age, BMI, acclimatisation, metabolic rate, hydration, HR trajectory) — deliberately excluding weather variables so a persona reflects *who the worker is*, not the day's forecast.

| Persona | Profile |
|---|---|
| Acclimatised Veteran | Experienced, high heat tolerance |
| Fit Young Worker | Physically capable, moderate exposure |
| High-Exertion Risk | Heavy workload driving risk |
| High-BMI Novice | Least acclimatised, most vulnerable |

Each persona gets a data-driven **adaptive alert multiplier** (0.80×–1.15×) that shifts their personal alert threshold earlier or later than baseline.
</details>

<details>
<summary><b>Phase 3 — XGBoost Model Training</b></summary><br>

- **SMOTE** oversampling (training set only) to handle the natural class imbalance
- **Optuna** Bayesian hyperparameter search (30 trials) with hard safety-recall constraints
- **5-fold stratified cross-validation** for robust performance estimates
- **SHAP TreeExplainer** for full model interpretability, per-class and per-prediction
</details>

<details>
<summary><b>Phase 4 — FastAPI Backend</b></summary><br>

Full inference pipeline replicating PHS feature computation, cluster assignment, and adaptive thresholding live at request time — not just a thin wrapper around `model.predict()`. Endpoints: `/predict`, `/predict/batch`, `/onboard`, `/model/info`, `/demo/scenario/{name}`. Covered by 48 tests spanning unit correctness, safety logic, and edge cases.
</details>

<details>
<summary><b>Phase 5 — App Prototype & Deployment</b></summary><br>

A fully interactive single-file HTML app (`worker_app.html`) with 5 screens — Onboarding, Dashboard, History, Profile, Supervisor View — driven by real model outputs. Model exported to ONNX for offline on-device inference via `onnxruntime-react-native`.
</details>

---

## 🎮 Live Demo

Open [`outputs/phase5/worker_app.html`](outputs/phase5/worker_app.html) in any browser — no server, no install. Four scenarios are pre-loaded with real model predictions:

| Scenario | Result |
|---|---|
| 🟢 Safe Worker (veteran, mild conditions) | Low risk, no alert |
| 🔴 Danger Worker (novice, extreme heat) | Critical, alert fires in 3 min |
| 🟡 Moderate Risk | Borderline, monitor closely |
| ⚠️ Early Warning (rising HR trajectory) | Catches danger before it's visible in current readings |

---

## 🧪 Testing

```bash
cd app/tests
python test_api.py
```

48 tests across 5 categories — inference engine unit tests, safety constraint validation, API endpoint behaviour, demo scenario regression checks, and edge cases (extremes, missing data, boundary values). All passing.

---

## 🩺 Model Card & Limitations

Full model card available at [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), including intended use, training data provenance, per-class metrics, and documented limitations. In short:

- Training data is **synthetic**, grounded in ISO 7933 but not yet field-validated
- K-Means clusters have moderate overlap (Silhouette ≈ 0.16) — expected given continuous physiological variation
- Accelerometer-based HR estimation on real phones will be noisier than the simulated Gaussian model

Field validation with real outdoor workers is the top priority for the next phase.

---

## 🗺️ Roadmap

- [ ] Field pilot with 50–100 real outdoor workers (construction / agriculture)
- [ ] Fine-tune model on real physiological data, validate cluster personas against outcomes
- [ ] React Native app using the exported ONNX model for offline inference
- [ ] SMS/supervisor-alert fallback for workers without smartphones
- [ ] Extend simulation framework to other high-heat-exposure regions (Gulf states, sub-Saharan Africa, SE Asia)

---

## 🌐 SDG Alignment

This project supports **UN SDG 3.9** (reduce deaths from environmental hazards) and **SDG 8.8** (protect labour rights and promote safe working environments), shifting occupational heat safety from reactive treatment to proactive, data-driven prevention.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details. Data generation pipeline, trained models, and all code are open for reuse, adaptation, and extension.

---

<div align="center">

**Built for the millions of outdoor workers who keep the world running through the hottest days on record.**

⭐ Star this repo if you'd like to see occupational heat safety made accessible everywhere.

</div>
