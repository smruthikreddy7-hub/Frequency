import unittest
import json
from app import create_app
from database import Database
from baseline_engine import BaselineEngine
from schema import ContextData, BaselineData, FrequencyInput
from prompt_builder import PromptBuilder

class Phase4And5TestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_database_context_save_and_retrieve(self):
        """Verify SQLite context save and get operations."""
        saved = Database.save_context(
            user_id="test_user",
            sleep_hours=5.5,
            sleep_quality="restless",
            workload="heavy",
            mood_score=4,
            notes="Late night deadline",
            target_date="2026-08-22"
        )
        self.assertEqual(saved["sleep_hours"], 5.5)
        self.assertEqual(saved["workload"], "heavy")
        self.assertEqual(saved["mood_score"], 4)

        retrieved = Database.get_context(user_id="test_user", target_date="2026-08-22")
        self.assertEqual(retrieved["sleep_quality"], "restless")

    def test_baseline_engine_calculation(self):
        """Verify BaselineEngine computes averages from history."""
        base = BaselineEngine.calculate_baseline(user_id="default_user")
        self.assertIsInstance(base, BaselineData)
        self.assertGreater(base.baseline_sleep_hours, 6.0)
        self.assertGreater(base.baseline_vocal_energy_rms, 0.01)
        self.assertGreater(base.baseline_speech_rate_wpm, 100.0)

    def test_baseline_deviations_and_correlations(self):
        """Verify deviation computation and cross-modal correlation detection."""
        context = {
            "user_id": "default_user",
            "sleep_hours": 4.5,  # ~40% below 7.5h baseline
            "workload": "heavy",
            "mood_score": 4
        }
        voice = {
            "vocal_energy_rms": 0.025,  # ~35% below 0.038 baseline
            "speech_rate_wpm": 120.0
        }
        text = {
            "sentiment": "negative",
            "emotional_valence": -0.6
        }

        result = BaselineEngine.compute_deviations_and_correlations(
            context_data=context,
            text_data=text,
            voice_data=voice
        )

        self.assertIn("sleep_hours_pct", result.deviations)
        self.assertLess(result.deviations["sleep_hours_pct"], -20.0)
        self.assertIn("vocal_energy_pct", result.deviations)
        self.assertLess(result.deviations["vocal_energy_pct"], -15.0)

        # Verify correlations identified
        self.assertIn("cross_modal_correlations", result.deviations)
        corrs = result.deviations["cross_modal_correlations"]
        self.assertTrue(any("Sleep deficit" in c for c in corrs))
        self.assertTrue(any("HEAVY" in c for c in corrs))

    def test_context_api_endpoints(self):
        """Verify GET and POST /api/context endpoints."""
        # GET
        get_res = self.client.get('/api/context')
        self.assertEqual(get_res.status_code, 200)
        self.assertIn("context", get_res.get_json())

        # POST
        payload = {
            "sleep_hours": 8.5,
            "workload": "light",
            "mood_score": 9
        }
        post_res = self.client.post(
            '/api/context',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(post_res.status_code, 200)
        data = post_res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["context"]["sleep_hours"], 8.5)

    def test_baseline_api_endpoint(self):
        """Verify GET /api/baseline endpoint."""
        res = self.client.get('/api/baseline')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("baseline", data)
        self.assertIn("baseline_sleep_hours", data["baseline"])

    def test_prompt_builder_includes_context_and_baseline(self):
        """Verify PromptBuilder formats context and baseline shifts into prompt."""
        pkg = FrequencyInput(
            user_prompt="Explain why I feel fatigued.",
            contextual_data={
                "sleep_hours": 4.5,
                "workload": "heavy"
            },
            baseline_data={
                "deviations": {
                    "sleep_hours_pct": -35.0,
                    "cross_modal_correlations": [
                        "Sleep deficit correlates with subdued vocal energy"
                    ]
                }
            }
        )
        compiled = PromptBuilder.build_evidence_prompt(pkg)
        self.assertIn("User-Reported Context", compiled)
        self.assertIn("Sleep Hours: 4.5", compiled)
        self.assertIn("Workload: heavy", compiled)
        self.assertIn("Personal Baselines & Deviations", compiled)
        self.assertIn("Sleep Hours Pct: -35.0", compiled)

if __name__ == '__main__':
    unittest.main()
