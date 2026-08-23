import numpy as np
from typing import Dict, Any, List, Optional, Union
from database import Database
from schema import BaselineData, ContextData, TextSignalData, VoiceSignalData

class BaselineEngine:
    """
    Personal Baseline & Cross-Modal Correlation Engine for FREQUENCY (Phase 5).
    Computes historical personal distributions, percentage shifts (Δ%), and multi-sense correlations.
    """

    @classmethod
    def calculate_baseline(cls, user_id: str = "default_user") -> BaselineData:
        """
        Calculate rolling personal baseline metrics from historical records.
        """
        context_rows = Database.get_historical_context(user_id=user_id, limit=14)
        session_rows = Database.get_historical_sessions(user_id=user_id, limit=30)

        # 1. Context Baselines
        sleep_vals = [r["sleep_hours"] for r in context_rows if r.get("sleep_hours") is not None]
        mood_vals = [r["mood_score"] for r in context_rows if r.get("mood_score") is not None]

        base_sleep = round(float(np.mean(sleep_vals)), 1) if sleep_vals else 7.5
        base_mood = round(float(np.mean(mood_vals)), 1) if mood_vals else 7.0

        # 2. Voice Baselines
        energy_vals = [r["vocal_energy_rms"] for r in session_rows if r.get("vocal_energy_rms") is not None and r["vocal_energy_rms"] > 0]
        wpm_vals = [r["speech_rate_wpm"] for r in session_rows if r.get("speech_rate_wpm") is not None and r["speech_rate_wpm"] > 0]
        pitch_vals = [r["mean_pitch_hz"] for r in session_rows if r.get("mean_pitch_hz") is not None and r["mean_pitch_hz"] > 0]

        base_energy = round(float(np.mean(energy_vals)), 4) if energy_vals else 0.038
        base_wpm = round(float(np.mean(wpm_vals)), 1) if wpm_vals else 145.0
        base_pitch = round(float(np.mean(pitch_vals)), 1) if pitch_vals else 128.0

        return BaselineData(
            user_id=user_id,
            baseline_sleep_hours=base_sleep,
            baseline_speech_rate_wpm=base_wpm,
            baseline_vocal_energy_rms=base_energy,
            baseline_mood_score=base_mood,
            deviations={}
        )

    @classmethod
    def compute_deviations_and_correlations(
        cls,
        context_data: Union[Dict[str, Any], ContextData],
        text_data: Optional[Union[Dict[str, Any], TextSignalData, str]] = None,
        voice_data: Optional[Union[Dict[str, Any], VoiceSignalData]] = None,
        baseline: Optional[BaselineData] = None
    ) -> BaselineData:
        """
        Compute percentage deviations (Δ%) and identify cross-modal correlations.
        """
        ctx_dict = context_data.to_dict() if hasattr(context_data, "to_dict") else (context_data if isinstance(context_data, dict) else {})
        text_dict = text_data.to_dict() if hasattr(text_data, "to_dict") else (text_data if isinstance(text_data, dict) else {})
        voice_dict = voice_data.to_dict() if hasattr(voice_data, "to_dict") else (voice_data if isinstance(voice_data, dict) else {})

        base = baseline or cls.calculate_baseline(user_id=ctx_dict.get("user_id", "default_user"))
        deviations = {}
        correlations = []

        # 1. Sleep Deviation
        current_sleep = ctx_dict.get("sleep_hours")
        if current_sleep is not None and base.baseline_sleep_hours:
            diff = current_sleep - base.baseline_sleep_hours
            pct = round((diff / base.baseline_sleep_hours) * 100, 1)
            deviations["sleep_hours_pct"] = pct

        # 2. Mood Deviation
        current_mood = ctx_dict.get("mood_score")
        if current_mood is not None and base.baseline_mood_score:
            diff = current_mood - base.baseline_mood_score
            pct = round((diff / base.baseline_mood_score) * 100, 1)
            deviations["mood_score_pct"] = pct

        # 3. Vocal Energy Deviation
        if voice_dict and voice_dict.get("vocal_energy_rms") is not None and base.baseline_vocal_energy_rms:
            curr_rms = voice_dict["vocal_energy_rms"]
            diff = curr_rms - base.baseline_vocal_energy_rms
            pct = round((diff / base.baseline_vocal_energy_rms) * 100, 1)
            deviations["vocal_energy_pct"] = pct

        # 4. Speech Rate Deviation
        if voice_dict and voice_dict.get("speech_rate_wpm") is not None and base.baseline_speech_rate_wpm:
            curr_wpm = voice_dict["speech_rate_wpm"]
            diff = curr_wpm - base.baseline_speech_rate_wpm
            pct = round((diff / base.baseline_speech_rate_wpm) * 100, 1)
            deviations["speech_rate_pct"] = pct

        # Cross-Modal Correlation Checks
        sleep_pct = deviations.get("sleep_hours_pct", 0)
        energy_pct = deviations.get("vocal_energy_pct", 0)
        speech_pct = deviations.get("speech_rate_pct", 0)
        
        if sleep_pct <= -20 and (energy_pct <= -15 or speech_pct <= -15):
            correlations.append(
                f"Sleep deficit ({sleep_pct}% vs baseline) strongly correlates with subdued vocal energy ({energy_pct}%) and slowed speaking pace."
            )
        elif sleep_pct >= 10 and energy_pct >= 10:
            correlations.append(
                f"Elevated sleep recovery (+{sleep_pct}% vs baseline) aligns with robust vocal energy (+{energy_pct}%)."
            )

        workload = ctx_dict.get("workload", "normal")
        sentiment = text_dict.get("sentiment") if text_dict else None

        if workload in ["heavy", "burnout"] and sentiment == "negative":
            correlations.append(
                f"Reported {workload.upper()} workload correlates with negative linguistic sentiment and elevated mental friction."
            )

        valence = text_dict.get("emotional_valence") if text_dict else None
        if current_mood is not None and valence is not None:
            if current_mood <= 4 and valence <= -0.3:
                correlations.append(
                    "Self-reported low mood aligns consistently with negative linguistic sentiment."
                )
            elif current_mood >= 8 and valence <= -0.3:
                correlations.append(
                    f"Possible Masked Strain: High self-reported mood ({current_mood}/10) contrasts with negative text sentiment ({valence}), suggesting compensatory effort."
                )

        base.deviations = deviations
        if correlations:
            base.deviations["cross_modal_correlations"] = correlations

        return base
