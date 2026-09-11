import unittest
import json
from app import create_app
from schema import FrequencyInput
from prompt_builder import PromptBuilder

class Phase1DTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_system_prompt_boundaries(self):
        """Verify the system prompt contains mandatory boundaries and role definition."""
        system_prompt = PromptBuilder.build_system_prompt()
        self.assertIn("FREQUENCY", system_prompt)
        self.assertIn("NON-DIAGNOSTIC", system_prompt)

    def test_evidence_prompt_compilation_conversational(self):
        """Verify compilation of a purely conversational message (lean prompt)."""
        pkg = FrequencyInput(user_prompt="Hey, how are you?")
        compiled = PromptBuilder.build_evidence_prompt(pkg)
        self.assertEqual(compiled, "Hey, how are you?")

    def test_evidence_prompt_compilation_multimodal(self):
        """Verify compilation of a rich multimodal package with sensory sections."""
        pkg = FrequencyInput(
            user_prompt="Analyze my energy state today.",
            text_data="Work was exhausting and I struggled to focus.",
            voice_data={
                "mean_pitch_hz": 125.4,
                "speech_rate_wpm": 165.0,
                "vocal_energy_rms": 0.015
            },
            contextual_data={
                "sleep_hours": 4.8,
                "workload": "heavy",
                "self_reported_mood": 3
            },
            baseline_data={
                "deviations": {
                    "vocal_energy_pct": -25.0,
                    "sleep_hours_pct": -32.0
                }
            }
        )
        compiled = PromptBuilder.build_evidence_prompt(pkg)

        self.assertIn("Analyze my energy state today.", compiled)
        self.assertIn("Work was exhausting and I struggled to focus.", compiled)
        self.assertIn("Mean Pitch Hz: 125.4", compiled)
        self.assertIn("Speech Rate Wpm: 165.0", compiled)
        self.assertIn("Sleep Hours: 4.8", compiled)
        self.assertIn("Workload: heavy", compiled)
        self.assertIn("Vocal Energy Pct: -25.0", compiled)

    def test_preview_endpoint(self):
        """Verify POST /api/prompt/preview endpoint."""
        payload = {
            "user_prompt": "Testing prompt preview",
            "contextual_data": {"sleep_hours": 7.5}
        }
        response = self.client.post(
            '/api/prompt/preview',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("evidence_prompt", data)
        self.assertIn("system_prompt", data)
        self.assertIn("active_modalities", data)
        self.assertIn("Sleep Hours: 7.5", data["evidence_prompt"])

    def test_streaming_with_prompt_builder(self):
        """Verify streaming chat leverages PromptBuilder seamlessly."""
        payload = {
            "user_prompt": "Say hello in 3 words."
        }
        response = self.client.post(
            '/api/chat/stream',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith('text/event-stream'))

if __name__ == '__main__':
    unittest.main()
