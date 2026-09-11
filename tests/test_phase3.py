import io
import unittest
import numpy as np
import soundfile as sf
from app import create_app
from voice_analyzer import VoiceAnalyzer
from schema import VoiceSignalData, FrequencyInput
from prompt_builder import PromptBuilder

class Phase3TestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.samplerate = 16000

    def _create_synthetic_wav(self, freq=150.0, duration=1.0, add_silence=False) -> bytes:
        """Helper to create synthetic WAV audio in memory."""
        t = np.linspace(0, duration, int(self.samplerate * duration), endpoint=False)
        # Sine wave tone in vocal range
        samples = 0.5 * np.sin(2 * np.pi * freq * t)
        
        if add_silence:
            # Add 0.5s of silence
            silence = np.zeros(int(self.samplerate * 0.5), dtype=np.float32)
            samples = np.concatenate([samples, silence])

        bio = io.BytesIO()
        sf.write(bio, samples.astype(np.float32), self.samplerate, format='WAV')
        return bio.getvalue()

    def test_extract_energy(self):
        """Verify RMS energy and peak amplitude computation."""
        wav_bytes = self._create_synthetic_wav(freq=180.0, duration=0.8)
        samples, sr = VoiceAnalyzer.load_audio(wav_bytes)
        energy = VoiceAnalyzer.extract_energy(samples)
        
        self.assertIn("vocal_energy_rms", energy)
        self.assertGreater(energy["vocal_energy_rms"], 0.2)
        self.assertGreater(energy["peak_amplitude"], 0.4)

    def test_extract_pitch(self):
        """Verify fundamental frequency F0 autocorrelation tracking."""
        target_f0 = 160.0
        wav_bytes = self._create_synthetic_wav(freq=target_f0, duration=1.0)
        samples, sr = VoiceAnalyzer.load_audio(wav_bytes)
        pitch = VoiceAnalyzer.extract_pitch(samples, sr)
        
        self.assertIn("mean_pitch_hz", pitch)
        # Pitch should be within 10% of target 160 Hz
        self.assertAlmostEqual(pitch["mean_pitch_hz"], target_f0, delta=20.0)

    def test_extract_temporal_metrics(self):
        """Verify duration, pause ratio, and speech rate metrics."""
        wav_bytes = self._create_synthetic_wav(freq=150.0, duration=1.0, add_silence=True)
        samples, sr = VoiceAnalyzer.load_audio(wav_bytes)
        metrics = VoiceAnalyzer.extract_temporal_metrics(samples, sr, word_count=5)
        
        self.assertGreater(metrics["audio_duration_seconds"], 1.3)
        self.assertGreater(metrics["pause_duration_ratio"], 0.1)
        self.assertGreater(metrics["speech_rate_wpm"], 50.0)

    def test_analyze_audio_data_pipeline(self):
        """Verify full analyze_audio_data pipeline with client transcript."""
        wav_bytes = self._create_synthetic_wav(freq=140.0, duration=1.0)
        voice_signals, transcript = VoiceAnalyzer.analyze_audio_data(
            audio_bytes=wav_bytes,
            filename="test_note.wav",
            client_transcript="Testing voice input"
        )
        self.assertIsInstance(voice_signals, VoiceSignalData)
        self.assertEqual(transcript, "Testing voice input")
        self.assertGreater(voice_signals.vocal_energy_rms, 0.1)

    def test_analyze_voice_api_endpoint(self):
        """Verify POST /api/analyze/voice multipart endpoint."""
        wav_bytes = self._create_synthetic_wav(freq=150.0, duration=0.8)
        data = {
            'file': (io.BytesIO(wav_bytes), 'recording.wav'),
            'transcript': 'I am feeling quite drained and sleepy'
        }
        response = self.client.post(
            '/api/analyze/voice',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertEqual(res_data["status"], "success")
        self.assertIn("voice_signals", res_data)
        self.assertIn("text_signals", res_data)
        self.assertEqual(res_data["text_signals"]["sentiment"], "negative")

    def test_prompt_builder_includes_voice_and_text(self):
        """Verify PromptBuilder compiles both voice acoustics and text into evidence."""
        voice_signals = VoiceSignalData(
            mean_pitch_hz=132.5,
            vocal_energy_rms=0.035,
            speech_rate_wpm=145.0,
            pause_duration_ratio=0.25
        )
        pkg = FrequencyInput(
            user_prompt="I am feeling overwhelmed today.",
            voice_data=voice_signals
        )
        compiled = PromptBuilder.build_evidence_prompt(pkg)
        self.assertIn("Extracted Voice & Acoustic Signals", compiled)
        self.assertIn("Mean Pitch Hz: 132.5", compiled)
        self.assertIn("Speech Rate Wpm: 145.0", compiled)

if __name__ == '__main__':
    unittest.main()
