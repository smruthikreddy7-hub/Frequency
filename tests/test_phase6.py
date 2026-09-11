import unittest
import json
from app import create_app
from fusion_engine import FusionEngine
from schema import FrequencyInput, FusionData, ContextData, TextSignalData, VoiceSignalData
from prompt_builder import PromptBuilder

class Phase6TestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_fusion_compute_energy_index_high_flow(self):
        """Verify high energy index on restful sleep, robust voice, and positive valence."""
        context = {"sleep_hours": 8.0, "mood_score": 8}
        voice = {"vocal_energy_rms": 0.042, "speech_rate_wpm": 145.0}
        text = {"emotional_valence": 0.6}

        energy = FusionEngine.compute_energy_index(context, voice, text)
        self.assertGreater(energy, 80.0)

    def test_fusion_compute_energy_index_sleep_deficit(self):
        """Verify low energy index on acute sleep deficit and subdued voice."""
        context = {"sleep_hours": 4.0, "mood_score": 4}
        voice = {"vocal_energy_rms": 0.012, "speech_rate_wpm": 110.0}
        text = {"emotional_valence": -0.5}

        energy = FusionEngine.compute_energy_index(context, voice, text)
        self.assertLess(energy, 50.0)

    def test_fusion_alignment_masking_penalty(self):
        """Verify alignment score detects masking mismatch (high mood vs negative text)."""
        context = {"mood_score": 9}
        text = {"emotional_valence": -0.6}
        voice = {"vocal_energy_rms": 0.010}

        align = FusionEngine.compute_alignment_score(context, voice, text)
        self.assertLess(align, 60.0)

    def test_fusion_archetype_classification(self):
        """Verify archetype classification logic."""
        # 1. Sleep deficit
        archetype, desc = FusionEngine.classify_archetype(
            energy_idx=42.0, focus_idx=55.0, alignment_score=85.0,
            context={"sleep_hours": 4.5, "workload": "normal"}
        )
        self.assertIn("Sleep-Deprived", archetype)

        # 2. Masked strain
        archetype_masked, _ = FusionEngine.classify_archetype(
            energy_idx=60.0, focus_idx=65.0, alignment_score=50.0,
            context={"sleep_hours": 7.5, "workload": "normal"}
        )
        self.assertIn("Masked Strain", archetype_masked)

    def test_fusion_api_endpoint(self):
        """Verify POST /api/fusion/synthesize endpoint."""
        payload = {
            "user_prompt": "Testing fusion synthesis",
            "contextual_data": {"sleep_hours": 7.5, "workload": "normal", "mood_score": 8}
        }
        res = self.client.post(
            '/api/fusion/synthesize',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("fusion", data)
        self.assertIn("energy_index", data["fusion"])
        self.assertIn("archetype", data["fusion"])

    def test_trends_api_endpoint(self):
        """Verify GET /api/trends returns 7-day trend history."""
        res = self.client.get('/api/trends')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("trends", data)
        self.assertGreaterEqual(len(data["trends"]), 1)

    def test_prompt_builder_includes_fusion_matrix(self):
        """Verify PromptBuilder compiles Cross-Sense Fusion into evidence prompt."""
        fusion = FusionData(
            energy_index=68.5,
            focus_index=74.0,
            alignment_score=90.0,
            archetype="Steady Baseline Equilibrium"
        )
        pkg = FrequencyInput(
            user_prompt="How is my overall focus today?",
            fusion_data=fusion
        )
        compiled = PromptBuilder.build_evidence_prompt(pkg)
        self.assertIn("Cross-Sense Fusion Synthesis & Archetype", compiled)
        self.assertIn("Energy Index: 68.5", compiled)
        self.assertIn("Steady Baseline Equilibrium", compiled)

if __name__ == '__main__':
    unittest.main()
