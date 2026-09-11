import io
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Any, Tuple, Optional
from schema import VoiceSignalData
from faster_whisper import WhisperModel

class VoiceAnalyzer:
    """
    Local acoustic signal extractor and speech transcription engine for FREQUENCY (Phase 3).
    Extracts pitch (F0), vocal energy (RMS), speech rate, and pause ratios.
    """

    MIN_VOICE_HZ = 75.0   # Low human vocal range
    MAX_VOICE_HZ = 450.0  # High human vocal range

    @classmethod
    def load_audio(cls, audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[np.ndarray, int]:
        """
        Load audio bytes using soundfile into a mono float32 numpy array and sample rate.
        """
        bio = io.BytesIO(audio_bytes)
        try:
            samples, samplerate = sf.read(bio, dtype='float32')
        except Exception as e:
            suffix = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                samples, samplerate = sf.read(tmp_path, dtype='float32')
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        return samples, samplerate

    @classmethod
    def extract_energy(cls, samples: np.ndarray) -> Dict[str, float]:
        """Compute RMS vocal energy and peak amplitude."""
        if len(samples) == 0:
            return {"vocal_energy_rms": 0.0, "peak_amplitude": 0.0, "dynamic_range_db": 0.0}

        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        
        if rms > 1e-6 and peak > 1e-6:
            dyn_range = float(20 * np.log10(peak / rms))
        else:
            dyn_range = 0.0

        return {
            "vocal_energy_rms": round(rms, 4),
            "peak_amplitude": round(peak, 4),
            "dynamic_range_db": round(dyn_range, 1)
        }

    @classmethod
    def extract_pitch(cls, samples: np.ndarray, samplerate: int) -> Dict[str, float]:
        """
        Autocorrelation-based fundamental frequency (F0) tracking within human vocal range.
        """
        if len(samples) < samplerate * 0.05:
            return {"mean_pitch_hz": 0.0, "pitch_variation_stdev": 0.0, "pitch_min_hz": 0.0, "pitch_max_hz": 0.0}

        frame_size = int(0.04 * samplerate)
        hop_size = int(0.015 * samplerate)
        
        min_lag = int(samplerate / cls.MAX_VOICE_HZ)
        max_lag = int(samplerate / cls.MIN_VOICE_HZ)

        if max_lag >= frame_size:
            frame_size = max_lag + 10

        f0_estimates = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start:start + frame_size]
            frame_rms = np.sqrt(np.mean(frame ** 2))
            if frame_rms < 0.01:
                continue

            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            if len(corr) <= max_lag:
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region) + min_lag
            if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.35:
                f0 = samplerate / peak_idx
                if cls.MIN_VOICE_HZ <= f0 <= cls.MAX_VOICE_HZ:
                    f0_estimates.append(f0)

        if not f0_estimates:
            return {
                "mean_pitch_hz": 120.0,
                "pitch_variation_stdev": 10.0,
                "pitch_min_hz": 110.0,
                "pitch_max_hz": 130.0,
                "voiced_ratio": 0.0
            }

        mean_f0 = float(np.mean(f0_estimates))
        stdev_f0 = float(np.std(f0_estimates))
        min_f0 = float(np.min(f0_estimates))
        max_f0 = float(np.max(f0_estimates))
        total_frames = max(1, (len(samples) - frame_size) // hop_size)
        voiced_ratio = round(len(f0_estimates) / total_frames, 2)

        return {
            "mean_pitch_hz": round(mean_f0, 1),
            "pitch_variation_stdev": round(stdev_f0, 1),
            "pitch_min_hz": round(min_f0, 1),
            "pitch_max_hz": round(max_f0, 1),
            "voiced_ratio": voiced_ratio
        }

    @classmethod
    def extract_temporal_metrics(cls, samples: np.ndarray, samplerate: int, word_count: int = 0) -> Dict[str, float]:
        """Compute audio duration, silence/pause duration ratio, and speaking rate (WPM)."""
        duration = float(len(samples) / samplerate)
        if duration <= 0:
            return {"audio_duration_seconds": 0.0, "pause_duration_ratio": 0.0, "speech_rate_wpm": 0.0}

        frame_size = int(0.03 * samplerate)
        hop_size = int(0.015 * samplerate)
        frame_energies = []

        for start in range(0, len(samples) - frame_size, hop_size):
            frame = samples[start:start + frame_size]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))

        if not frame_energies:
            return {"audio_duration_seconds": round(duration, 2), "pause_duration_ratio": 0.0, "speech_rate_wpm": 0.0}

        max_energy = max(frame_energies) if frame_energies else 1.0
        silence_threshold = max(0.005, max_energy * 0.12)
        silence_frames = sum(1 for e in frame_energies if e < silence_threshold)
        pause_ratio = round(silence_frames / len(frame_energies), 2)

        speaking_time_mins = max(0.02, (duration * (1.0 - pause_ratio)) / 60.0)
        if word_count > 0:
            wpm = round(word_count / speaking_time_mins, 1)
        else:
            peaks = sum(1 for i in range(1, len(frame_energies) - 1) if frame_energies[i] > frame_energies[i-1] and frame_energies[i] > frame_energies[i+1] and frame_energies[i] > silence_threshold)
            est_words = max(1, int(peaks * 0.6))
            wpm = round(est_words / max(0.02, duration / 60.0), 1)

        return {
            "audio_duration_seconds": round(duration, 2),
            "pause_duration_ratio": pause_ratio,
            "speech_rate_wpm": min(300.0, max(50.0, wpm))
        }

    @classmethod
    def transcribe(cls, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribe audio locally using Whisper.
        No audio is sent to an external speech-recognition service.
        """
        try:
            samples, samplerate = cls.load_audio(audio_bytes, filename)

            # Create a temporary WAV file for Whisper
            suffix = os.path.splitext(filename)[1] or ".wav"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as tmp:
                temp_path = tmp.name

            try:
                sf.write(
                    temp_path,
                    samples,
                    samplerate,
                    subtype="PCM_16"
                )

                model = cls.get_whisper_model()

                segments, info = model.transcribe(
                    temp_path,
                    beam_size=5,
                    vad_filter=True
                )

                transcript = " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                )

                return transcript.strip()

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            print(f"Local Whisper transcription error: {e}")
            return ""
        
    @classmethod
    def analyze_audio_data(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        client_transcript: Optional[str] = None
    ) -> Tuple[VoiceSignalData, str]:
        """
        Full voice signal extraction pipeline.
        Returns populated VoiceSignalData and the transcript text.
        """
        samples, samplerate = cls.load_audio(audio_bytes, filename)
        
        if client_transcript and client_transcript.strip():
            transcript = client_transcript.strip()
        else:
            transcript = cls.transcribe(audio_bytes, filename)
            if not transcript:
                transcript = "Voice note shared."

        word_count = len(transcript.split()) if transcript else 0

        energy_metrics = cls.extract_energy(samples)
        pitch_metrics = cls.extract_pitch(samples, samplerate)
        temporal_metrics = cls.extract_temporal_metrics(samples, samplerate, word_count=word_count)

        voice_signals = VoiceSignalData(
            audio_duration_seconds=temporal_metrics["audio_duration_seconds"],
            speech_rate_wpm=temporal_metrics["speech_rate_wpm"],
            mean_pitch_hz=pitch_metrics["mean_pitch_hz"],
            pitch_variation_stdev=pitch_metrics["pitch_variation_stdev"],
            vocal_energy_rms=energy_metrics["vocal_energy_rms"],
            pause_duration_ratio=temporal_metrics["pause_duration_ratio"],
            transcript=transcript,
            confidence=0.9 if energy_metrics["vocal_energy_rms"] > 0.01 else 0.5
        )

        return voice_signals, transcript
    _whisper_model = None

    @classmethod
    def get_whisper_model(cls):
        """Load Whisper locally once and reuse it."""
        if cls._whisper_model is None:
            model_path = os.environ.get(
                "WHISPER_MODEL_PATH",
                "models/whisper-base"
            )

            cls._whisper_model = WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8"
            )

        return cls._whisper_model
