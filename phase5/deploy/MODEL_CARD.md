# HeatGuard Model Card
## Model: XGBoost Heatstroke Risk Classifier v1.0

---

### Model Details

| Field | Value |
|---|---|
| Model type | XGBoost Classifier (multi:softprob) |
| Task | 4-class risk classification |
| Classes | Low (0), Moderate (1), High (2), Critical (3) |
| Features | 27 engineered features |
| Training date | March 2026 |
| Version | 1.0.0 |

**Best hyperparameters (Optuna, 30 trials):**
- `n_estimators`: 485
- `max_depth`: 9
- `learning_rate`: 0.1447
- `subsample`: 0.6253
- `colsample_bytree`: 0.5605
- `min_child_weight`: 2
- `gamma`: 0.0429
- `reg_alpha`: 0.33
- `reg_lambda`: 2.0443

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

**Features used (27):**
- `age`
- `bmi`
- `acclimatisation_days`
- `ambient_temp`
- `humidity`
- `wind_speed`
- `solar_radiation`
- `ambient_temp_t15`
- `ambient_temp_t30`
- `humidity_t15`
- `humidity_t30`
- `metabolic_rate`
- `work_hours`
- `hydration_level`
- `heart_rate`
- `heart_rate_t15`
- `heart_rate_t30`
- `sweat_rate`
- `core_temp_tre`
- `heat_index`
- `hr_delta_t15`
- `hr_delta_t30`
- `temp_delta_t15`
- `temp_humidity_product`
- `vulnerability_score`
- `cluster_id`
- `adaptive_alert_multiplier`

---

### Performance Metrics (Test Set — 15% held-out, never seen during training)

| Metric | Value |
|---|---|
| **F1-macro** | **0.9533** ← primary metric |
| ROC-AUC macro | 0.9982 |
| Accuracy | 0.9741 (not headline — see below) |

**Per-class metrics:**

| Class | Sensitivity (Recall) | F1-score | AUC | Precision |
|---|---|---|---|---|
| Low | 0.9667 | 0.9748 | 1.0 | 0.9831 |
| Moderate | 0.95 | 0.9421 | 0.9994 | 0.9344 |
| High | 0.94 | 0.9082 | 0.9953 | 0.8785 |
| Critical | 0.9825 | 0.988 | 0.9984 | 0.9936 |

**Why not accuracy?** The dataset has class imbalance (60.7% Critical). A model
predicting Critical for every worker achieves 60.7% accuracy — meaningless.
Sensitivity (Recall) for High and Critical classes is the primary metric because
a false negative (missed collapse) is categorically more dangerous than a false
positive (unnecessary rest break).

**5-fold cross-validation:**
- F1-macro: 0.9470 ± 0.0100
- Recall(High): 0.9058 ± 0.0208
- Recall(Critical): 0.9836 ± 0.0032

---

### Safety Constraints

The model was tuned with hard safety constraints:
- `Recall(High) >= 0.75` — **achieved: 0.94** ✓
- `Recall(Critical) >= 0.80` — **achieved: 0.9825** ✓

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

*Generated: 2026-04-02 10:14 UTC*
*HeatGuard v1.0 — Open-source. MIT License.*
