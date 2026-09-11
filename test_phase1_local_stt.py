"""
FREQUENCY - Phase 1 Local STT Test
==================================

Run after replacing Google Speech Recognition with local faster-whisper.

Basic test:
    python test_phase1_local_stt.py

Test with a real voice recording:
    python test_phase1_local_stt.py path/to/voice.wav

Expected:
    PASS - required local dependencies/imports
    PASS - no Google/cloud SpeechRecognition references
    PASS - local acoustic analysis
    PASS - local Whisper transcription (when a real audio file is supplied)

If no audio file is supplied, the Whisper test is skipped because synthetic
tones cannot produce meaningful speech transcription.
"""

import sys
import os
import inspect
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
VOICE_ANALYZER = ROOT / "voice_analyzer.py"


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    print(f"[FAIL] {message}")
    return False


def test_source_is_local():
    source = VOICE_ANALYZER.read_text(encoding="utf-8")

    ok = True

    forbidden = [
        "recognize_google",
        "speech_recognition",
        "import speech_recognition",
        "from speech_recognition",
    ]

    for item in forbidden:
        if item in source:
            print(f"[FAIL] Cloud/Google STT reference still exists: {item}")
            ok = False

    if ok:
        print("[PASS] No Google/SpeechRecognition STT reference found")

    check(
        "faster_whisper" in source or "WhisperModel" in source,
        "Local Whisper implementation is present",
    )

    return ok


def test_import():
    try:
        from voice_analyzer import VoiceAnalyzer
        print("[PASS] VoiceAnalyzer imports successfully")
        return VoiceAnalyzer
    except Exception as exc:
        print(f"[FAIL] VoiceAnalyzer import failed: {exc}")
        return None


def test_acoustic_pipeline(VoiceAnalyzer):
    """
    Generate a short synthetic vocal-like signal and verify that the existing
    local acoustic analysis still works.
    """
    sample_rate = 16000
    duration = 2.0

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False,
    )

    # Synthetic 180 Hz tone with a small amplitude modulation.
    samples = (
        0.12
        * (0.8 + 0.2 * np.sin(2 * np.pi * 4 * t))
        * np.sin(2 * np.pi * 180 * t)
    ).astype(np.float32)

    wav_buffer = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    wav_path = wav_buffer.name
    wav_buffer.close()

    try:
        sf.write(wav_path, samples, sample_rate)

        audio_bytes = Path(wav_path).read_bytes()

        voice_signals, transcript = VoiceAnalyzer.analyze_audio_data(
            audio_bytes,
            filename="phase1_synthetic.wav",
            client_transcript="This is a Phase 1 local processing test.",
        )

        ok = True
        ok &= check(
            voice_signals.audio_duration_seconds > 0,
            "Audio duration extracted locally",
        )
        ok &= check(
            voice_signals.vocal_energy_rms > 0,
            "Vocal energy extracted locally",
        )
        ok &= check(
            voice_signals.mean_pitch_hz > 0,
            "Pitch extracted locally",
        )
        ok &= check(
            voice_signals.speech_rate_wpm > 0,
            "Speech-rate metric calculated locally",
        )
        ok &= check(
            transcript == "This is a Phase 1 local processing test.",
            "Client transcript flows through the voice pipeline",
        )

        return ok

    except Exception as exc:
        print(f"[FAIL] Acoustic pipeline failed: {exc}")
        return False

    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def test_local_whisper(VoiceAnalyzer, audio_path):
    """
    Run actual local transcription against a real speech recording.

    The audio file must contain spoken words. This test intentionally does not
    use Google's API or any other remote STT service.
    """
    if not audio_path:
        print("[SKIP] Whisper transcription test")
        print("       Supply a real speech WAV/audio file to run it.")
        return True

    path = Path(audio_path)

    if not path.exists():
        print(f"[FAIL] Audio file not found: {path}")
        return False

    try:
        source = inspect.getsource(VoiceAnalyzer.transcribe)

        if "recognize_google" in source:
            print("[FAIL] transcribe() still contains recognize_google()")
            return False

        if "faster_whisper" not in Path(
            VOICE_ANALYZER
        ).read_text(encoding="utf-8") and "WhisperModel" not in source:
            print("[FAIL] transcribe() does not appear to use local Whisper")
            return False

        audio_bytes = path.read_bytes()

        print(f"[INFO] Transcribing locally: {path}")
        transcript = VoiceAnalyzer.transcribe(
            audio_bytes,
            filename=path.name,
        )

        print(f"[INFO] Transcript: {transcript!r}")

        return check(
            bool(transcript.strip()),
            "Local Whisper returned a non-empty transcript",
        )

    except Exception as exc:
        print(f"[FAIL] Local Whisper transcription failed: {exc}")
        print("       Check that faster-whisper is installed and the local")
        print("       Whisper model is available.")
        return False


def main():
    print("=" * 60)
    print("FREQUENCY - PHASE 1 LOCAL STT TEST")
    print("=" * 60)

    all_passed = True

    print("\n1. Checking source code...")
    all_passed &= test_source_is_local()

    print("\n2. Checking Python import...")
    VoiceAnalyzer = test_import()

    if VoiceAnalyzer is None:
        all_passed = False
    else:
        print("\n3. Testing local acoustic pipeline...")
        all_passed &= test_acoustic_pipeline(VoiceAnalyzer)

        audio_path = sys.argv[1] if len(sys.argv) > 1 else None

        print("\n4. Testing local Whisper STT...")
        all_passed &= test_local_whisper(VoiceAnalyzer, audio_path)

    print("\n" + "=" * 60)

    if all_passed:
        print("PHASE 1 TEST RESULT: PASS")
        print("Voice processing is ready for the next phase.")
        return 0

    print("PHASE 1 TEST RESULT: FAIL")
    print("Fix the failed checks before moving to Phase 2.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
