"""
=============================================================================
PHASE 4: TEST SUITE
=============================================================================
Comprehensive automated tests for the inference engine and API.
All tests run without a live server (uses FastAPI TestClient).

Test categories:
    1. InferenceEngine unit tests (direct function calls)
    2. API endpoint tests (via TestClient — no server needed)
    3. Demo scenario tests (regression checks)
    4. Safety constraint tests (alert logic)
    5. Edge case / boundary tests
    6. Batch prediction tests
    7. Onboarding tests

Run:
    python test_api.py
    python test_api.py -v           (verbose)
    python test_api.py TestSafety   (single class)
=============================================================================
"""

import json
import sys
import unittest
from pathlib import Path

# Add parent API dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from fastapi.testclient import TestClient
from app import app
from inference_engine import (
    predict,
    compute_heat_index,
    compute_core_temp_proxy,
    compute_vulnerability_score,
    compute_adaptive_multiplier,
    RISK_LABELS,
)

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

SAFE_WORKER = {
    "age": 32.0, "bmi": 22.5, "acclimatisation_days": 65.0,
    "ambient_temp": 31.0, "humidity": 45.0, "wind_speed": 2.5,
    "solar_radiation": 320.0, "metabolic_rate": 155.0,
    "work_hours": 2.0, "hydration_level": 4.0,
    "heart_rate": 82.0,
    "ambient_temp_t15": 30.2, "ambient_temp_t30": 29.5,
    "humidity_t15": 44.0, "humidity_t30": 43.5,
    "heart_rate_t15": 80.5, "heart_rate_t30": 79.0,
}

DANGER_WORKER = {
    "age": 48.0, "bmi": 31.0, "acclimatisation_days": 6.0,
    "ambient_temp": 44.5, "humidity": 87.0, "wind_speed": 0.3,
    "solar_radiation": 920.0, "metabolic_rate": 355.0,
    "work_hours": 6.5, "hydration_level": 1.0,
    "heart_rate": 138.0,
    "ambient_temp_t15": 43.5, "ambient_temp_t30": 42.8,
    "humidity_t15": 85.0, "humidity_t30": 83.5,
    "heart_rate_t15": 130.0, "heart_rate_t30": 122.0,
}

MINIMAL_WORKER = {
    "age": 30.0, "bmi": 23.0, "acclimatisation_days": 30.0,
    "ambient_temp": 35.0, "humidity": 60.0,
    "metabolic_rate": 200.0, "work_hours": 3.0,
    "hydration_level": 3.0, "heart_rate": 95.0,
    # No lag values — should default gracefully
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. INFERENCE ENGINE UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceEngine(unittest.TestCase):

    def test_heat_index_formula_extremes(self):
        """Heat index should be clipped between 25 and 70."""
        hi_low  = compute_heat_index(20.0, 10.0)
        hi_high = compute_heat_index(50.0, 100.0)
        self.assertGreaterEqual(hi_low,  25.0)
        self.assertLessEqual(hi_high, 70.0)

    def test_heat_index_increases_with_temp(self):
        """Higher temperature must produce higher heat index."""
        hi_30 = compute_heat_index(30.0, 60.0)
        hi_45 = compute_heat_index(45.0, 60.0)
        self.assertGreater(hi_45, hi_30)

    def test_heat_index_increases_with_humidity(self):
        """Higher humidity must produce higher heat index."""
        hi_40 = compute_heat_index(38.0, 40.0)
        hi_90 = compute_heat_index(38.0, 90.0)
        self.assertGreater(hi_90, hi_40)

    def test_core_temp_within_physiological_range(self):
        """PHS core temp must always be clipped to [36.0, 39.5]."""
        tre = compute_core_temp_proxy(
            ambient_temp=50.0, humidity=100.0, wind_speed=0.0,
            solar_radiation=1200.0, metabolic_rate=500.0, work_hours=12.0,
            hydration_level=1.0, acclimatisation_days=0.0, bmi=40.0, age=60.0
        )
        self.assertGreaterEqual(tre, 36.0)
        self.assertLessEqual(tre, 39.5)

    def test_core_temp_low_for_safe_conditions(self):
        """Safe conditions should produce low core temp."""
        tre = compute_core_temp_proxy(
            ambient_temp=28.0, humidity=30.0, wind_speed=3.0,
            solar_radiation=100.0, metabolic_rate=100.0, work_hours=1.0,
            hydration_level=5.0, acclimatisation_days=90.0, bmi=20.0, age=25.0
        )
        self.assertLess(tre, 38.0)

    def test_vulnerability_score_direction(self):
        """High BMI/low acclimatisation must score higher than fit veteran."""
        novice  = {"bmi": 35.0, "age": 50.0, "acclimatisation_days": 5.0,
                   "metabolic_rate": 380.0, "hydration_level": 1.0, "hr_delta_t30": 15.0}
        veteran = {"bmi": 20.0, "age": 28.0, "acclimatisation_days": 80.0,
                   "metabolic_rate": 130.0, "hydration_level": 5.0, "hr_delta_t30": 1.0}
        self.assertGreater(
            compute_vulnerability_score(novice),
            compute_vulnerability_score(veteran)
        )

    def test_adaptive_multiplier_range(self):
        """Multiplier must be in [0.70, 1.20] for any vulnerability score."""
        for vuln in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            mult = compute_adaptive_multiplier(vuln)
            self.assertGreaterEqual(mult, 0.70,
                msg=f"Multiplier below 0.70 for vuln={vuln}")
            self.assertLessEqual(mult, 1.20,
                msg=f"Multiplier above 1.20 for vuln={vuln}")

    def test_adaptive_multiplier_direction(self):
        """Higher vulnerability → lower multiplier (earlier alert)."""
        mult_low_vuln  = compute_adaptive_multiplier(-0.5)
        mult_high_vuln = compute_adaptive_multiplier(0.5)
        self.assertGreater(mult_low_vuln, mult_high_vuln)

    def test_predict_returns_required_keys(self):
        """Prediction response must contain all required keys."""
        result   = predict(SAFE_WORKER)
        required = [
            "predicted_class", "predicted_class_num", "probabilities",
            "risk_score", "alert_fires", "threshold_used",
            "adaptive_alert_multiplier", "cluster_id", "persona_name",
            "vulnerability_score", "core_temp_estimate", "heat_index",
            "time_to_peak_minutes", "risk_color", "risk_emoji",
            "risk_message", "risk_action", "top_risk_factors",
        ]
        for key in required:
            self.assertIn(key, result, msg=f"Missing key: {key}")

    def test_predict_probabilities_sum_to_one(self):
        """Class probabilities must sum to 1.0 (± floating point tolerance)."""
        result = predict(SAFE_WORKER)
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_predict_class_matches_argmax(self):
        """predicted_class must be the label with highest probability."""
        result  = predict(DANGER_WORKER)
        max_label = max(result["probabilities"], key=result["probabilities"].get)
        self.assertEqual(result["predicted_class"], max_label)

    def test_risk_score_range(self):
        """Risk score must be in [0, 100]."""
        for worker in [SAFE_WORKER, DANGER_WORKER, MINIMAL_WORKER]:
            result = predict(worker)
            self.assertGreaterEqual(result["risk_score"], 0.0)
            self.assertLessEqual(result["risk_score"], 100.0)

    def test_minimal_worker_no_lag_fields(self):
        """Inference must succeed with only required fields (no lag values)."""
        try:
            result = predict(MINIMAL_WORKER)
            self.assertIn("predicted_class", result)
        except Exception as e:
            self.fail(f"predict() raised {e} with minimal worker input")

    def test_time_to_peak_positive(self):
        """time_to_peak_minutes must always be a positive integer."""
        for worker in [SAFE_WORKER, DANGER_WORKER]:
            result = predict(worker)
            self.assertGreater(result["time_to_peak_minutes"], 0)

    def test_top_risk_factors_max_three(self):
        """top_risk_factors must return at most 3 items."""
        result = predict(DANGER_WORKER)
        self.assertLessEqual(len(result["top_risk_factors"]), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAFETY CONSTRAINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyConstraints(unittest.TestCase):

    def test_danger_worker_fires_alert(self):
        """The canonical danger scenario must fire an alert."""
        result = predict(DANGER_WORKER)
        self.assertTrue(result["alert_fires"],
            msg=f"Alert should fire for danger worker. Got: {result['predicted_class']}")

    def test_safe_worker_does_not_fire_alert(self):
        """The canonical safe scenario must not fire an alert."""
        result = predict(SAFE_WORKER)
        self.assertFalse(result["alert_fires"],
            msg=f"Alert should NOT fire for safe worker. Got: {result['predicted_class']}")

    def test_critical_worker_high_risk_score(self):
        """Critical workers must have risk_score above 70."""
        result = predict(DANGER_WORKER)
        self.assertGreater(result["risk_score"], 70.0,
            msg=f"Danger worker risk_score too low: {result['risk_score']}")

    def test_safe_worker_low_risk_score(self):
        """Safe workers must have risk_score below 40."""
        result = predict(SAFE_WORKER)
        self.assertLess(result["risk_score"], 40.0,
            msg=f"Safe worker risk_score too high: {result['risk_score']}")

    def test_adaptive_threshold_shifts_per_persona(self):
        """Higher vulnerability worker must have lower effective threshold."""
        safe_result   = predict(SAFE_WORKER)
        danger_result = predict(DANGER_WORKER)
        # Lower multiplier → lower threshold → alert fires earlier
        self.assertGreaterEqual(
            safe_result["adaptive_alert_multiplier"],
            danger_result["adaptive_alert_multiplier"],
            msg="Safe worker should have >= multiplier than danger worker"
        )

    def test_time_to_peak_lower_for_danger(self):
        """Danger worker must have shorter time-to-peak than safe worker."""
        safe_ttp   = predict(SAFE_WORKER)["time_to_peak_minutes"]
        danger_ttp = predict(DANGER_WORKER)["time_to_peak_minutes"]
        self.assertLess(danger_ttp, safe_ttp,
            msg=f"Danger TTP ({danger_ttp}) should be < Safe TTP ({safe_ttp})")

    def test_hr_trajectory_captured(self):
        """hr_trajectory must reflect difference between current and t-30 HR."""
        result       = predict(DANGER_WORKER)
        expected_traj = DANGER_WORKER["heart_rate"] - DANGER_WORKER["heart_rate_t30"]
        self.assertAlmostEqual(result["hr_trajectory"], expected_traj, places=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. API ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints(unittest.TestCase):

    def test_health_check_200(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "healthy")
        self.assertTrue(r.json()["models_loaded"])

    def test_health_check_n_features(self):
        r = client.get("/health")
        self.assertEqual(r.json()["n_features"], 27)

    def test_model_info_200(self):
        r = client.get("/model/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("features",       data)
        self.assertIn("phase3_metrics", data)
        self.assertEqual(len(data["features"]), 27)

    def test_predict_endpoint_safe_worker(self):
        r = client.post("/predict", json=SAFE_WORKER)
        self.assertEqual(r.status_code, 200)
        result = r.json()
        self.assertIn("predicted_class", result)
        self.assertIn(result["predicted_class"], RISK_LABELS)

    def test_predict_endpoint_danger_worker(self):
        r = client.post("/predict", json=DANGER_WORKER)
        self.assertEqual(r.status_code, 200)
        result = r.json()
        self.assertTrue(result["alert_fires"])

    def test_predict_endpoint_minimal_worker(self):
        """Endpoint must accept minimal input (no lag fields)."""
        r = client.post("/predict", json=MINIMAL_WORKER)
        self.assertEqual(r.status_code, 200)

    def test_predict_invalid_hydration(self):
        """Invalid hydration_level must return 422 Unprocessable Entity."""
        bad_worker = {**SAFE_WORKER, "hydration_level": 6.0}
        r = client.post("/predict", json=bad_worker)
        self.assertEqual(r.status_code, 422)

    def test_predict_out_of_range_temp(self):
        """Temperature below 20°C must return 422."""
        bad_worker = {**SAFE_WORKER, "ambient_temp": 10.0}
        r = client.post("/predict", json=bad_worker)
        self.assertEqual(r.status_code, 422)

    def test_predict_missing_required_field(self):
        """Missing required field must return 422."""
        bad_worker = {k: v for k, v in SAFE_WORKER.items() if k != "heart_rate"}
        r = client.post("/predict", json=bad_worker)
        self.assertEqual(r.status_code, 422)

    def test_predict_response_schema(self):
        """All required response fields must be present and correct types."""
        r      = client.post("/predict", json=SAFE_WORKER)
        result = r.json()
        self.assertIsInstance(result["predicted_class"],    str)
        self.assertIsInstance(result["predicted_class_num"], int)
        self.assertIsInstance(result["probabilities"],      dict)
        self.assertIsInstance(result["risk_score"],         float)
        self.assertIsInstance(result["alert_fires"],        bool)
        self.assertIsInstance(result["cluster_id"],         int)
        self.assertIsInstance(result["time_to_peak_minutes"], int)

    def test_batch_predict_200(self):
        """Batch endpoint must return results for all workers."""
        batch_payload = {"workers": [SAFE_WORKER, DANGER_WORKER, MINIMAL_WORKER]}
        r = client.post("/predict/batch", json=batch_payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["total_workers"], 3)
        self.assertEqual(len(data["results"]),  3)
        self.assertIn("alert_count",    data)
        self.assertIn("critical_count", data)
        self.assertIn("processing_ms",  data)

    def test_batch_alert_count_consistent(self):
        """alert_count must match sum of individual alert_fires."""
        batch_payload = {"workers": [SAFE_WORKER, DANGER_WORKER]}
        r    = client.post("/predict/batch", json=batch_payload)
        data = r.json()
        expected = sum(1 for res in data["results"] if res["alert_fires"])
        self.assertEqual(data["alert_count"], expected)

    def test_onboard_endpoint_200(self):
        """Onboarding endpoint must return cluster assignment."""
        onboard_data = {
            "age": 34.0, "bmi": 24.5, "acclimatisation_days": 12.0,
            "metabolic_rate": 250.0, "hydration_level": 3.0,
        }
        r = client.post("/onboard", json=onboard_data)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("cluster_id",               data)
        self.assertIn("persona_name",             data)
        self.assertIn("vulnerability_score",      data)
        self.assertIn("adaptive_alert_multiplier", data)
        self.assertIn("personalisation_summary",  data)

    def test_onboard_returns_valid_cluster_id(self):
        """Cluster ID must be within [0, n_clusters-1]."""
        from inference_engine import get_registry
        n_clusters = get_registry().kmeans.n_clusters
        onboard_data = {"age": 30.0, "bmi": 22.0, "acclimatisation_days": 30.0}
        r = client.post("/onboard", json=onboard_data)
        self.assertIn(r.json()["cluster_id"], range(n_clusters))


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEMO SCENARIO TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoScenarios(unittest.TestCase):

    def test_safe_scenario_loads(self):
        r = client.get("/demo/scenario/safe")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("prediction", data)
        self.assertFalse(data["prediction"]["alert_fires"],
            msg="Safe scenario should not fire alert")

    def test_danger_scenario_fires_alert(self):
        r = client.get("/demo/scenario/danger")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["prediction"]["alert_fires"],
            msg="Danger scenario must fire alert")

    def test_all_scenarios_return_200(self):
        for name in ["safe", "danger", "moderate", "critical_early"]:
            r = client.get(f"/demo/scenario/{name}")
            self.assertEqual(r.status_code, 200,
                msg=f"Scenario '{name}' returned {r.status_code}")

    def test_invalid_scenario_404(self):
        r = client.get("/demo/scenario/nonexistent")
        self.assertEqual(r.status_code, 404)

    def test_critical_early_trajectory_detected(self):
        """The critical_early scenario must show rising HR trajectory."""
        r    = client.get("/demo/scenario/critical_early")
        data = r.json()["prediction"]
        self.assertGreater(data["hr_trajectory"], 5.0,
            msg="critical_early scenario must show significant HR rise")

    def test_danger_has_risk_factors(self):
        """Danger scenario must return at least one risk factor."""
        r    = client.get("/demo/scenario/danger")
        data = r.json()["prediction"]
        self.assertGreater(len(data["top_risk_factors"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. EDGE CASE / BOUNDARY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_maximum_stress_conditions(self):
        """Maximum physiological stress must not crash the API."""
        extreme = {
            "age": 60.0, "bmi": 40.0, "acclimatisation_days": 0.0,
            "ambient_temp": 54.9, "humidity": 99.9, "wind_speed": 0.0,
            "solar_radiation": 1199.9, "metabolic_rate": 499.9,
            "work_hours": 11.9, "hydration_level": 1.0,
            "heart_rate": 199.9,
            "ambient_temp_t15": 53.0, "ambient_temp_t30": 51.0,
            "humidity_t15": 98.0, "humidity_t30": 96.0,
            "heart_rate_t15": 185.0, "heart_rate_t30": 170.0,
        }
        r = client.post("/predict", json=extreme)
        self.assertEqual(r.status_code, 200)

    def test_minimum_stress_conditions(self):
        """Minimum physiological stress must not crash the API."""
        minimal_stress = {
            "age": 20.0, "bmi": 18.0, "acclimatisation_days": 90.0,
            "ambient_temp": 20.0, "humidity": 20.0, "wind_speed": 10.0,
            "solar_radiation": 10.0, "metabolic_rate": 100.0,
            "work_hours": 0.1, "hydration_level": 5.0,
            "heart_rate": 60.0,
        }
        r = client.post("/predict", json=minimal_stress)
        self.assertEqual(r.status_code, 200)

    def test_lag_values_same_as_current(self):
        """Identical lag values (static conditions) must not crash."""
        static = {**SAFE_WORKER}
        static["ambient_temp_t15"] = static["ambient_temp"]
        static["ambient_temp_t30"] = static["ambient_temp"]
        static["heart_rate_t15"]   = static["heart_rate"]
        static["heart_rate_t30"]   = static["heart_rate"]
        r = client.post("/predict", json=static)
        self.assertEqual(r.status_code, 200)

    def test_falling_heart_rate_negative_delta(self):
        """Falling HR (worker cooling down) must produce negative hr_trajectory."""
        cooling = {**SAFE_WORKER,
                   "heart_rate": 80.0,
                   "heart_rate_t15": 90.0,
                   "heart_rate_t30": 100.0}
        result = predict(cooling)
        self.assertLess(result["hr_trajectory"], 0.0)

    def test_probabilities_all_present(self):
        """All 4 risk class probabilities must be in the response."""
        result = predict(SAFE_WORKER)
        for label in RISK_LABELS:
            self.assertIn(label, result["probabilities"])
            self.assertGreaterEqual(result["probabilities"][label], 0.0)
            self.assertLessEqual(result["probabilities"][label],    1.0)

    def test_batch_empty_list_rejected(self):
        """Empty batch must return 422."""
        r = client.post("/predict/batch", json={"workers": []})
        self.assertIn(r.status_code, [422, 400])


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    """Run all tests and print a summary report."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestInferenceEngine,
        TestSafetyConstraints,
        TestAPIEndpoints,
        TestDemoScenarios,
        TestEdgeCases,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"  PHASE 4 TEST SUMMARY")
    print("=" * 60)
    print(f"  Tests run   : {result.testsRun}")
    print(f"  Passed      : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures    : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print(f"  Status      : {'✓ ALL PASSED' if result.wasSuccessful() else '✗ SOME FAILED'}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
