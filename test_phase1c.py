import unittest
import json
from app import create_app
from schema import FrequencyInput

class Phase1CTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_schema_dataclass_defaults_and_validation(self):
        """Verify FrequencyInput dataclass defaults and validation logic."""
        # Valid minimal
        pkg = FrequencyInput(user_prompt="Why do I feel tired?")
        is_valid, err = pkg.validate()
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(pkg.get_active_modalities(), ["prompt"])
        self.assertIsNone(pkg.voice_data)
        self.assertEqual(pkg.contextual_data, {})
        self.assertEqual(pkg.baseline_data, {})

        # Invalid empty prompt
        invalid_pkg = FrequencyInput(user_prompt="   ")
        is_valid, err = invalid_pkg.validate()
        self.assertFalse(is_valid)
        self.assertIn("user_prompt", err)

        # Modality detection
        full_pkg = FrequencyInput(
            user_prompt="Explain signals",
            text_data="I worked all day",
            voice_data={"speech_rate_wpm": 160},
            contextual_data={"sleep_hours": 5.5},
            baseline_data={"deviations": {"sleep": -20}}
        )
        modalities = full_pkg.get_active_modalities()
        self.assertEqual(set(modalities), {"prompt", "text", "voice", "context", "baseline"})

    def test_schema_serialization(self):
        """Verify serialization roundtrip to_dict and from_dict."""
        original = FrequencyInput(
            user_prompt="Test prompt",
            text_data="Sample raw text",
            contextual_data={"sleep_hours": 7.0}
        )
        dict_rep = original.to_dict()
        reconstructed = FrequencyInput.from_dict(dict_rep)

        self.assertEqual(original.user_prompt, reconstructed.user_prompt)
        self.assertEqual(original.text_data, reconstructed.text_data)
        self.assertEqual(original.contextual_data, reconstructed.contextual_data)

    def test_schema_endpoint(self):
        """Verify GET /api/schema returns JSON schema specification."""
        response = self.client.get('/api/schema')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("title"), "FrequencyInput")
        self.assertIn("properties", data)
        self.assertIn("user_prompt", data["properties"])
        self.assertIn("voice_data", data["properties"])
        self.assertIn("contextual_data", data["properties"])

    def test_package_validation_endpoint(self):
        """Verify POST /api/analyze/package validates packages."""
        # Valid package
        valid_payload = {
            "user_prompt": "Analyzing cross-sense patterns",
            "text_data": "Exhausted after meeting",
            "contextual_data": {"workload": "heavy"}
        }
        res_valid = self.client.post(
            '/api/analyze/package',
            data=json.dumps(valid_payload),
            content_type='application/json'
        )
        self.assertEqual(res_valid.status_code, 200)
        data = res_valid.get_json()
        self.assertEqual(data.get("status"), "valid")
        self.assertIn("prompt", data.get("active_modalities", []))
        self.assertIn("text", data.get("active_modalities", []))
        self.assertIn("context", data.get("active_modalities", []))

        # Invalid package
        invalid_payload = {"user_prompt": ""}
        res_invalid = self.client.post(
            '/api/analyze/package',
            data=json.dumps(invalid_payload),
            content_type='application/json'
        )
        self.assertEqual(res_invalid.status_code, 400)

    def test_streaming_with_structured_package(self):
        """Verify POST /api/chat/stream accepts structured FrequencyInput."""
        payload = {
            "user_prompt": "Say test ok in 2 words.",
            "text_data": "sample context",
            "contextual_data": {"sleep_hours": 8}
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
