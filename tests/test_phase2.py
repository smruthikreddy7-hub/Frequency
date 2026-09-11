import unittest
import json
from app import create_app
from text_analyzer import TextAnalyzer
from schema import TextSignalData, FrequencyInput
from prompt_builder import PromptBuilder

class Phase2TestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_sentiment_extraction_positive(self):
        """Verify sentiment extraction on positive, energized text."""
        text = "I feel rested, energized, and very productive today!"
        res = TextAnalyzer.extract_sentiment(text)
        self.assertEqual(res["polarity"], "positive")
        self.assertGreater(res["valence_score"], 0.3)

    def test_sentiment_extraction_negative(self):
        """Verify sentiment extraction on fatigue and stress text."""
        text = "Work was exhausting, I feel completely drained and overwhelmed."
        res = TextAnalyzer.extract_sentiment(text)
        self.assertEqual(res["polarity"], "negative")
        self.assertLess(res["valence_score"], -0.4)

    def test_sentiment_negation_handling(self):
        """Verify negation inversion (e.g. 'not happy')."""
        text = "I am not happy and barely focused."
        res = TextAnalyzer.extract_sentiment(text)
        self.assertEqual(res["polarity"], "negative")

    def test_emotion_detection(self):
        """Verify emotion categories are detected."""
        text = "I am exhausted and sleepy after a frantic deadline."
        emotions = TextAnalyzer.extract_emotions(text)
        self.assertIn("exhaustion_fatigue", emotions)
        self.assertIn("stress_anxiety", emotions)
        self.assertGreater(emotions["exhaustion_fatigue"], 0.3)

    def test_theme_classification(self):
        """Verify thematic keyword classification."""
        text = "My code project has a tight deadline, and I only got 4 hours of sleep."
        themes = TextAnalyzer.extract_themes(text)
        self.assertIn("workload_career", themes)
        self.assertIn("sleep_recovery", themes)

    def test_linguistic_metrics(self):
        """Verify linguistic characteristics and cognitive load."""
        text = "I guess maybe I'm confused about why my focus is scattered today?"
        metrics = TextAnalyzer.extract_linguistic_metrics(text)
        self.assertGreater(metrics["word_count"], 5)
        self.assertGreaterEqual(metrics["hesitation_markers"], 2)
        self.assertEqual(metrics["question_count"], 1)

    def test_full_text_analyzer_pipeline(self):
        """Verify full analyze method returns populated TextSignalData."""
        text = "Completely exhausted from work meetings and struggling to concentrate."
        signals = TextAnalyzer.analyze(text)
        self.assertIsInstance(signals, TextSignalData)
        self.assertEqual(signals.sentiment, "negative")
        self.assertIn("exhaustion_fatigue", signals.emotional_signals)
        self.assertIn("workload_career", signals.detected_themes)
        self.assertIn("focus_productivity", signals.detected_themes)

    def test_analyze_text_api_endpoint(self):
        """Verify POST /api/analyze/text endpoint."""
        payload = {"text": "I feel wonderful and motivated!"}
        response = self.client.post(
            '/api/analyze/text',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("signals", data)
        self.assertEqual(data["signals"]["sentiment"], "positive")

    def test_prompt_builder_includes_text_signals(self):
        """Verify PromptBuilder formats TextSignalData into evidence section."""
        signals = TextAnalyzer.analyze("Exhausted from heavy workload.")
        pkg = FrequencyInput(
            user_prompt="Why am I sluggish?",
            text_data=signals
        )
        compiled = PromptBuilder.build_evidence_prompt(pkg)
        self.assertIn("Extracted Text & Linguistic Signals", compiled)
        self.assertIn("Sentiment: negative", compiled)
        self.assertIn("workload_career", compiled)

if __name__ == '__main__':
    unittest.main()
