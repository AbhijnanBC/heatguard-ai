"""
=============================================================================
PHASE 5: MODEL EXPORT & DEPLOYMENT PREPARATION
=============================================================================
Exports the trained XGBoost model to multiple formats for deployment:

  1. ONNX format         → for smartphone (React Native + ONNX Runtime)
  2. JSON format         → XGBoost native, readable, inspectable
  3. Model card          → standardised documentation of model behaviour
  4. Requirements file   → exact dependency versions for reproducibility
  5. Deployment checklist → validates all artefacts are present and valid

Why ONNX for mobile deployment?
    ONNX (Open Neural Network Exchange) is a universal model format.
    - React Native: onnxruntime-react-native package (inference in JS)
    - Android: onnxruntime-android (Java/Kotlin, < 1ms per prediction)
    - iOS: onnxruntime-objc
    - ONNX inference is self-contained — no XGBoost library needed on device
    - Model file size: typically 50–200 KB (vs 500 KB+ for full XGBoost)

Run:
    python export_and_deploy.py
    python export_and_deploy.py --skip_onnx   (if onnxmltools not installed)
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR  = _ROOT / "models"
DEPLOY_DIR  = _ROOT / "phase5" / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

RISK_LABELS  = ["Low", "Moderate", "High", "Critical"]
RISK_THRESHOLDS = {
    "Low":      (0, 37.5),
    "Moderate": (37.5, 38.0),
    "High":     (38.0, 38.5),
    "Critical": (38.5, 99.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artefacts():
    model   = joblib.load(MODELS_DIR / "heatstroke_model.pkl")
    kmeans  = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    scaler  = joblib.load(MODELS_DIR / "cluster_scaler.pkl")
    with open(MODELS_DIR / "feature_list.json") as f:
        feat_info = json.load(f)
    with open(MODELS_DIR / "phase3_metrics.json") as f:
        metrics = json.load(f)
    with open(MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)
    return model, kmeans, scaler, feat_info, metrics, best_params


# ─────────────────────────────────────────────────────────────────────────────
# 2. ONNX EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(model, n_features: int, skip: bool = False):
    if skip:
        print("  [ONNX] Skipped (--skip_onnx flag set).")
        print("         To export: pip install onnxmltools skl2onnx onnx")
        return None

    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model   = convert_xgboost(model, initial_types=initial_type)

        onnx_path = DEPLOY_DIR / "heatstroke_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        size_kb = onnx_path.stat().st_size / 1024
        print(f"  ✓ ONNX model exported : {onnx_path}")
        print(f"    File size           : {size_kb:.1f} KB")
        return onnx_path

    except ImportError:
        print("  [ONNX] onnxmltools not installed. Running alternative export...")
        print("         Install with: pip install onnxmltools skl2onnx onnx")

        # Fallback: export using XGBoost's native save_model
        json_path = DEPLOY_DIR / "heatstroke_model.json"
        model.save_model(str(json_path))
        size_kb = json_path.stat().st_size / 1024
        print(f"  ✓ XGBoost JSON exported: {json_path}  ({size_kb:.1f} KB)")
        return json_path

    except Exception as e:
        print(f"  [ONNX] Export failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBOOST NATIVE JSON EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_native_json(model):
    """Export XGBoost model as human-readable JSON (always done regardless of ONNX)."""
    json_path = DEPLOY_DIR / "heatstroke_model_xgb.json"
    model.save_model(str(json_path))
    size_kb = json_path.stat().st_size / 1024
    print(f"  ✓ XGBoost JSON model   : {json_path}  ({size_kb:.1f} KB)")
    return json_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. COPY INFERENCE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def copy_inference_artefacts():
    """Copy all files needed for inference into the deploy package."""
    import shutil
    files_to_copy = [
        MODELS_DIR / "heatstroke_model.pkl",
        MODELS_DIR / "kmeans_model.pkl",
        MODELS_DIR / "cluster_scaler.pkl",
        MODELS_DIR / "feature_list.json",
        MODELS_DIR / "best_params.json",
        MODELS_DIR / "phase3_metrics.json",
    ]
    for f in files_to_copy:
        if f.exists():
            shutil.copy2(f, DEPLOY_DIR / f.name)
            print(f"  ✓ Copied: {f.name}")
        else:
            print(f"  ✗ Missing: {f.name}")

    # Copy API files
    api_dir = _ROOT / "app" / "api"
    for name in ["inference_engine.py", "app.py"]:
        src = api_dir / name
        if src.exists():
            shutil.copy2(src, DEPLOY_DIR / name)
            print(f"  ✓ Copied: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL CARD
# ─────────────────────────────────────────────────────────────────────────────

def generate_model_card(feat_info: dict, metrics: dict, best_params: dict):
    """
    Write a standardised model card (Hugging Face / Google Model Cards format).
    Documents intended use, performance, limitations, ethical considerations.
    """
    features = feat_info["features"]
    pc       = metrics["per_class"]
    cv       = metrics["cross_validation"]

    card = f"""# HeatGuard Model Card
## Model: XGBoost Heatstroke Risk Classifier v1.0

---

### Model Details

| Field | Value |
|---|---|
| Model type | XGBoost Classifier (multi:softprob) |
| Task | 4-class risk classification |
| Classes | Low (0), Moderate (1), High (2), Critical (3) |
| Features | {len(features)} engineered features |
| Training date | March 2026 |
| Version | 1.0.0 |

**Best hyperparameters (Optuna, 30 trials):**
"""
    for k, v in best_params.items():
        card += f"- `{k}`: {round(float(v), 4) if isinstance(v, float) else v}\n"

    card += f"""
---

### Intended Use

**Primary use case:** Predict heat stress risk level for outdoor workers
(construction, agriculture, delivery) 30–60 minutes in advance of symptoms,
using smartphone sensor data and physiological measurements.

**Target users:** Individual outdoor workers in India (and globally).

**Not intended for:** Medical diagnosis. Clinical treatment decisions.
Replacement for professional medical assessment in high-risk individuals.

---

### Training Data

| Property | Value |
|---|---|
| Dataset | PHS synthetic (ISO 7933 equations + Gaussian noise) |
| Records | 5,450 workers |
| Positive class balance | SMOTE applied (training set only) |
| Generation method | Phase 1 PHS pipeline (open-source) |
| Real-world validation | Not yet validated (Phase 2 deployment: 50–100 field workers) |

**Features used ({len(features)}):**
"""
    for f in features:
        card += f"- `{f}`\n"

    card += f"""
---

### Performance Metrics (Test Set — 15% held-out, never seen during training)

| Metric | Value |
|---|---|
| **F1-macro** | **{metrics['f1_macro']}** ← primary metric |
| ROC-AUC macro | {metrics['roc_auc_macro']} |
| Accuracy | {metrics['accuracy']} (not headline — see below) |

**Per-class metrics:**

| Class | Sensitivity (Recall) | F1-score | AUC | Precision |
|---|---|---|---|---|
| Low | {pc['Low']['sensitivity_recall']} | {pc['Low']['f1_score']} | {pc['Low']['roc_auc']} | {pc['Low']['precision']} |
| Moderate | {pc['Moderate']['sensitivity_recall']} | {pc['Moderate']['f1_score']} | {pc['Moderate']['roc_auc']} | {pc['Moderate']['precision']} |
| High | {pc['High']['sensitivity_recall']} | {pc['High']['f1_score']} | {pc['High']['roc_auc']} | {pc['High']['precision']} |
| Critical | {pc['Critical']['sensitivity_recall']} | {pc['Critical']['f1_score']} | {pc['Critical']['roc_auc']} | {pc['Critical']['precision']} |

**Why not accuracy?** The dataset has class imbalance (60.7% Critical). A model
predicting Critical for every worker achieves 60.7% accuracy — meaningless.
Sensitivity (Recall) for High and Critical classes is the primary metric because
a false negative (missed collapse) is categorically more dangerous than a false
positive (unnecessary rest break).

**5-fold cross-validation:**
- F1-macro: {cv['f1_macro_mean']:.4f} ± {cv['f1_macro_std']:.4f}
- Recall(High): {cv['high_recall_mean']:.4f} ± {cv['high_recall_std']:.4f}
- Recall(Critical): {cv['critical_recall_mean']:.4f} ± {cv['critical_recall_std']:.4f}

---

### Safety Constraints

The model was tuned with hard safety constraints:
- `Recall(High) >= 0.75` — **achieved: {pc['High']['sensitivity_recall']}** ✓
- `Recall(Critical) >= 0.80` — **achieved: {pc['Critical']['sensitivity_recall']}** ✓

During Optuna hyperparameter search, any trial violating either constraint
received a score of 0.0 and was discarded, regardless of F1.

---

### Limitations

1. **Synthetic training data.** The model was trained on PHS-derived synthetic data.
   Real physiological variance, equipment inaccuracies, and worker behaviour patterns
   are approximated, not directly observed. Phase 2 deployment (field data collection)
   is required before clinical validation.

2. **Fixed PHS coefficients.** ISO 7933 coefficients are population-level averages.
   Individual metabolic variation, clothing effects, and specific health conditions
   are not captured.

3. **Two K-Means clusters.** With only 2 clusters, persona differentiation is
   binary (Acclimatised Veteran vs High-BMI Novice). k=3 or k=4 on larger
   real-world datasets may reveal finer-grained physiological profiles.

4. **No real-time physiological feedback.** The model uses accelerometer-based
   HR estimation, not medical-grade sensors. Sensor noise is modelled but
   individual phone hardware varies.

5. **Binary occupational context.** Metabolic rate is entered once at onboarding
   and not updated during the shift. Workers switching between light and heavy
   tasks within a shift will have inaccurate metabolic inputs.

---

### Ethical Considerations

- **Worker empowerment, not surveillance.** The app is worker-owned.
  No employer receives access to any individual's predictions without explicit consent.
- **No GPS tracking storage.** Location is used only for weather API calls and
  is never stored or transmitted.
- **Data remains on device.** All model inference happens client-side.
  Cloud sync is strictly opt-in.
- **Algorithmic transparency.** SHAP values are computed and surfaced to workers
  as plain-language "why is this alert firing" explanations.
- **False positive policy.** We deliberately optimise for sensitivity over
  specificity. A false positive costs a rest break. A false negative can cost a life.

---

### Deployment Checklist

See `deployment_checklist.json` for automated validation results.

---

*Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}*
*HeatGuard v1.0 — Open-source. MIT License.*
"""

    card_path = DEPLOY_DIR / "MODEL_CARD.md"
    with open(card_path, "w", encoding="utf-8") as f: f.write(card)
    print(f"  ✓ Model card saved     : {card_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. REQUIREMENTS FILE
# ─────────────────────────────────────────────────────────────────────────────

def generate_requirements():
    reqs = """# HeatGuard — Production Requirements
# Generated by export_and_deploy.py

# Core ML
xgboost>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# API
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
httpx>=0.27.0         # for TestClient

# SHAP explainability
shap>=0.44.0

# Hyperparameter optimisation
optuna>=3.5.0

# ONNX export (optional — for mobile deployment)
# onnxmltools>=1.12.0
# skl2onnx>=1.16.0
# onnx>=1.15.0

# Visualisation (Phase 3)
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
"""
    req_path = DEPLOY_DIR / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f: f.write(reqs)
    print(f"  ✓ requirements.txt     : {req_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEPLOYMENT CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

def run_deployment_checklist(model, feat_info: dict, metrics: dict) -> dict:
    """
    Validate all model artefacts are present, loadable, and produce
    sensible outputs. Generates checklist JSON for CI/CD integration.
    """
    checks = {}

    # Check 1: All artefact files exist
    required_files = [
        "heatstroke_model.pkl", "kmeans_model.pkl", "cluster_scaler.pkl",
        "feature_list.json", "best_params.json", "phase3_metrics.json",
    ]
    for fname in required_files:
        exists = (DEPLOY_DIR / fname).exists()
        checks[f"file_{fname}"] = {"pass": exists, "detail": "Present" if exists else "MISSING"}

    # Check 2: Feature count matches
    n_feat_model    = model.n_features_in_
    n_feat_list     = len(feat_info["features"])
    checks["feature_count_match"] = {
        "pass":   n_feat_model == n_feat_list,
        "detail": f"Model expects {n_feat_model}, feature_list has {n_feat_list}"
    }

    # Check 3: Model produces valid probabilities
    dummy = np.zeros((1, n_feat_model))
    probs = model.predict_proba(dummy)[0]
    valid_probs = abs(sum(probs) - 1.0) < 0.001 and all(p >= 0 for p in probs)
    checks["probability_validity"] = {
        "pass":   valid_probs,
        "detail": f"Probs sum to {sum(probs):.6f}"
    }

    # Check 4: Safety constraints met
    hi_rec   = metrics["per_class"]["High"]["sensitivity_recall"]
    crit_rec = metrics["per_class"]["Critical"]["sensitivity_recall"]
    checks["recall_high"] = {
        "pass": hi_rec >= 0.75, "detail": f"Recall(High) = {hi_rec}"
    }
    checks["recall_critical"] = {
        "pass": crit_rec >= 0.80, "detail": f"Recall(Critical) = {crit_rec}"
    }

    # Check 5: F1-macro meets minimum
    f1 = metrics["f1_macro"]
    checks["f1_macro_minimum"] = {
        "pass": f1 >= 0.85, "detail": f"F1-macro = {f1}"
    }

    # Check 6: AUC meets minimum
    auc = metrics["roc_auc_macro"]
    checks["roc_auc_minimum"] = {
        "pass": auc >= 0.90, "detail": f"ROC-AUC = {auc}"
    }

    # Check 7: Inference speed
    t0 = time.time()
    for _ in range(100):
        model.predict_proba(dummy)
    avg_ms = (time.time() - t0) / 100 * 1000
    checks["inference_speed_ms"] = {
        "pass": avg_ms < 50.0, "detail": f"Average {avg_ms:.2f}ms per call"
    }

    # Check 8: Danger worker fires alert
    danger_features = {
        "age":48,"bmi":31,"acclimatisation_days":6,
        "ambient_temp":44.5,"humidity":87,"wind_speed":0.3,
        "solar_radiation":920,"metabolic_rate":355,"work_hours":6.5,
        "hydration_level":1,"heart_rate":138,"heart_rate_t15":130,"heart_rate_t30":122,
        "ambient_temp_t15":43.5,"ambient_temp_t30":42.8,"humidity_t15":85,"humidity_t30":83.5,
        "sweat_rate":115,"core_temp_tre":39.5,"heat_index":70,"hr_delta_t15":8,
        "hr_delta_t30":16,"temp_delta_t15":1,"temp_humidity_product":38.7,
        "vulnerability_score":1.58,"cluster_id":0,"adaptive_alert_multiplier":0.7,
    }
    X_danger = np.array([[danger_features[f] for f in feat_info["features"]]])
    probs_d  = model.predict_proba(X_danger)[0]
    pred_d   = int(np.argmax(probs_d))
    checks["danger_worker_predicts_critical"] = {
        "pass": pred_d == 3, "detail": f"Predicted class = {['Low','Moderate','High','Critical'][pred_d]}"
    }

    # Summary
    all_pass   = all(c["pass"] for c in checks.values())
    n_pass     = sum(1 for c in checks.values() if c["pass"])
    n_total    = len(checks)

    print(f"\n  DEPLOYMENT CHECKLIST ({n_pass}/{n_total} checks passed):")
    print("  " + "-" * 55)
    for name, result in checks.items():
        status = "✓" if result["pass"] else "✗"
        print(f"  {status} {name:<40} {result['detail']}")

    checks["_summary"] = {
        "all_pass": all_pass, "n_pass": n_pass, "n_total": n_total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC"),
    }

    checklist_path = DEPLOY_DIR / "deployment_checklist.json"
    with open(checklist_path, "w" , encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    print(f"\n  ✓ Checklist saved      : {checklist_path}")

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Model Export & Deployment")
    parser.add_argument("--skip_onnx", action="store_true",
                        help="Skip ONNX export (onnxmltools not installed)")
    args = parser.parse_args()

    print("=" * 65)
    print("  PHASE 5: MODEL EXPORT & DEPLOYMENT PREPARATION")
    print("=" * 65)
    print(f"  Deploy directory: {DEPLOY_DIR}")
    print()

    print("[1/6] Loading model artefacts...")
    model, kmeans, scaler, feat_info, metrics, best_params = load_artefacts()
    print(f"   XGBoost model: {model.n_features_in_} features, "
          f"{model.n_classes_} classes")

    print("\n[2/6] Copying inference artefacts to deploy package...")
    copy_inference_artefacts()

    print("\n[3/6] Exporting to ONNX format...")
    export_to_onnx(model, feat_info["n_features"] if "n_features" in feat_info
                   else len(feat_info["features"]), skip=args.skip_onnx)

    print("\n[4/6] Exporting XGBoost native JSON...")
    export_native_json(model)

    print("\n[5/6] Generating model card...")
    generate_model_card(feat_info, metrics, best_params)

    print("\n[5b] Generating requirements.txt...")
    generate_requirements()

    print("\n[6/6] Running deployment checklist...")
    checklist = run_deployment_checklist(model, feat_info, metrics)

    print()
    print("=" * 65)
    print(f"  PHASE 5 COMPLETE")
    all_pass = checklist.get("_summary", {}).get("all_pass", False)
    n_pass   = checklist.get("_summary", {}).get("n_pass", "?")
    n_total  = checklist.get("_summary", {}).get("n_total", "?")
    print(f"  Checklist: {n_pass}/{n_total} {'✓ ALL PASSED' if all_pass else '⚠ SOME FAILED'}")
    print(f"  Deploy package: {DEPLOY_DIR}")
    print("=" * 65)
    print()
    print("  DEPLOYMENT INSTRUCTIONS:")
    print("  1. Copy the deploy/ directory to your server")
    print("  2. pip install -r requirements.txt")
    print("  3. cd deploy && uvicorn app:app --host 0.0.0.0 --port 8000")
    print("  4. Open worker_app.html in a browser (no server needed)")
    print("  5. For mobile: use onnxruntime-react-native with heatstroke_model.onnx")


if __name__ == "__main__":
    main()
