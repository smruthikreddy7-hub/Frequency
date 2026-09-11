from typing import Dict, Any, List, Optional, Tuple, Union
from schema import FusionData, ContextData, BaselineData, TextSignalData, VoiceSignalData, FrequencyInput

class FusionEngine:
    """
    Cross-Sense Fusion Engine & Matrix Synthesizer for FREQUENCY (Phase 6 Capstone).
    Synthesizes Text, Voice, Context, and Personal Baseline streams into unified composite scores,
    classifies holistic Pattern Archetypes, and generates actionable micro-recommendations.
    """

    @classmethod
    def compute_energy_index(
        cls,
        context: Dict[str, Any],
        voice: Optional[Dict[str, Any]] = None,
        text: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute normalized Energy Index (0.0 to 100.0%).
        Weights: Sleep (35%), Vocal RMS (25%), Speech Rate (20%), Text Valence (20%).
        """
        scores = []
        weights = []

        # Ensure baseline is always a dictionary
        baseline = baseline or {}

        # ============================================================
        # 1. Sleep Component — Personal Baseline (35%)
        # ============================================================
        sleep_hours = context.get("sleep_hours")
        baseline_sleep = baseline.get("baseline_sleep_hours")

        if sleep_hours is not None:
            if baseline_sleep is not None and baseline_sleep > 0:
                sleep_deviation = abs(
                    sleep_hours - baseline_sleep
                ) / baseline_sleep

                # 100 = exactly at personal baseline.
                # Larger deviation = lower score.
                s_score = max(
                    20.0,
                    min(100.0, 100.0 - (sleep_deviation * 200.0))
                )
            else:
                # Not enough personal history yet.
                s_score = 70.0

            scores.append(s_score)
            weights.append(0.35)

        # ============================================================
        # 2. Vocal Energy RMS — Personal Baseline (25%)
        # ============================================================
        if voice and voice.get("vocal_energy_rms") is not None:
            rms = voice["vocal_energy_rms"]
            baseline_rms = baseline.get("baseline_vocal_energy_rms")

            if baseline_rms is not None and baseline_rms > 0:
                rms_deviation = abs(
                    rms - baseline_rms
                ) / baseline_rms

                # 100 = normal for this user.
                v_score = max(
                    20.0,
                    min(100.0, 100.0 - (rms_deviation * 200.0))
                )
            else:
                v_score = 70.0

            scores.append(v_score)
            weights.append(0.25)

        # ============================================================
        # 3. Speech Rate — Personal Baseline (20%)
        # ============================================================
        if voice and voice.get("speech_rate_wpm") is not None:
            wpm = voice["speech_rate_wpm"]
            baseline_wpm = baseline.get("baseline_speech_rate_wpm")

            if baseline_wpm is not None and baseline_wpm > 0:
                wpm_deviation = abs(
                    wpm - baseline_wpm
                ) / baseline_wpm

                # 100 = normal speaking rate for this user.
                w_score = max(
                    20.0,
                    min(100.0, 100.0 - (wpm_deviation * 200.0))
                )
            else:
                w_score = 70.0

            scores.append(w_score)
            weights.append(0.20)

        # ============================================================
        # 4. Text Valence (20%)
        # ============================================================
        if text and text.get("emotional_valence") is not None:
            val = text["emotional_valence"]

            # Text valence is already normalized from -1.0 to +1.0.
            t_score = max(
                0.0,
                min(100.0, ((val + 1.0) / 2.0) * 100.0)
            )

            scores.append(t_score)
            weights.append(0.20)

        # ============================================================
        # 5. Mood Fallback — Personal Baseline
        # ============================================================
        elif context.get("mood_score") is not None:
            mood = context["mood_score"]
            baseline_mood = baseline.get("baseline_mood_score")

            if baseline_mood is not None and baseline_mood > 0:
                mood_deviation = abs(
                    mood - baseline_mood
                ) / baseline_mood

                # 100 = normal mood for this user.
                m_score = max(
                    20.0,
                    min(100.0, 100.0 - (mood_deviation * 200.0))
                )
            else:
                m_score = 70.0

            scores.append(m_score)
            weights.append(0.20)

        # ============================================================
        # Final weighted Energy Index
        # ============================================================
        if not scores:
            return 70.0

        total_weight = sum(weights)
        weighted_sum = sum(
            score * weight
            for score, weight in zip(scores, weights)
        )

        return round(weighted_sum / total_weight, 1)
    @classmethod
    def compute_focus_index(
        cls,
        context: Dict[str, Any],
        voice: Optional[Dict[str, Any]] = None,
        text: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute Cognitive Focus & Clarity Index (0.0 to 100.0%).
        Evaluates pause ratios, hesitation markers, and workload strain.
        """
        scores = []
        weights = []

        # 1. Voice Pause Ratio (Low pauses -> High focus flow)
        if voice and voice.get("pause_duration_ratio") is not None:
            pause_ratio = voice["pause_duration_ratio"]
            # 0% pause -> 100%, 35% pause -> 40%
            p_score = max(15.0, min(100.0, (1.0 - (pause_ratio / 0.5)) * 100.0))
            scores.append(p_score)
            weights.append(0.40)

        # 2. Text Cognitive Load & Hesitation
        if text:
            cog_load = text.get("cognitive_load", "moderate")
            if cog_load == "low":
                c_score = 90.0
            elif cog_load == "moderate":
                c_score = 75.0
            elif cog_load == "elevated":
                c_score = 55.0
            else:
                c_score = 40.0
            scores.append(c_score)
            weights.append(0.35)

        # 3. Workload Drag
        workload = context.get("workload", "normal")
        if workload == "light":
            w_score = 85.0
        elif workload == "normal":
            w_score = 80.0
        elif workload == "heavy":
            w_score = 60.0
        else:
            w_score = 40.0
        scores.append(w_score)
        weights.append(0.25)

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum / total_weight, 1)

    @classmethod
    def compute_alignment_score(
        cls,
        context: Dict[str, Any],
        voice: Optional[Dict[str, Any]] = None,
        text: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute Cross-Sense Multi-Modal Alignment Score (0.0 to 100.0%).
        Measures agreement vs contradiction across Text, Voice, and Context.
        """
        penalty = 0.0

        # Mismatch A: High reported mood vs negative text sentiment
        mood = context.get("mood_score", 7)
        valence = text.get("emotional_valence") if text else None
        if valence is not None:
            if mood >= 8 and valence <= -0.3:
                penalty += 30.0  # Severe masking mismatch
            elif mood <= 3 and valence >= 0.4:
                penalty += 25.0

        # Mismatch B: High reported mood vs very subdued vocal energy
        if voice and voice.get("vocal_energy_rms") is not None:
            rms = voice["vocal_energy_rms"]
            if mood >= 8 and rms < 0.015:
                penalty += 25.0  # Vocal exhaustion hidden behind high mood rating

        # Mismatch C: Low sleep but user claims light/happy without friction
        sleep = context.get("sleep_hours", 7.5)
        if sleep < 5.0 and valence is not None and valence > 0.6:
            penalty += 15.0  # Sleep debt compensation

        alignment = max(20.0, 100.0 - penalty)
        return round(alignment, 1)

    @classmethod
    def classify_archetype(
        cls,
        energy_idx: float,
        focus_idx: float,
        alignment_score: float,
        context: Dict[str, Any],
        voice: Optional[Dict[str, Any]] = None,
        text: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Classify the overarching multimodal pattern archetype.
        """
        sleep_hours = context.get("sleep_hours", 7.5)
        workload = context.get("workload", "normal")
        wpm = voice.get("speech_rate_wpm", 145.0) if voice else 145.0
        valence = text.get("emotional_valence", 0.0) if text else 0.0

        if alignment_score < 65.0:
            return (
                "Masked Strain & Compensatory Effort",
                "High self-reported expectations contrast with underlying acoustic drain and linguistic fatigue."
            )
        elif sleep_hours < 5.5 and energy_idx < 55.0:
            return (
                "Sleep-Deprived Cognitive Sluggishness",
                "Acute sleep deficit is depressing vocal amplitude and slowing mental processing speed."
            )
        elif workload in ["heavy", "burnout"] and wpm > 165 and valence < -0.2:
            return (
                "High-Arousal Task Pressure & Rush",
                "Workload urgency is creating rapid vocal pacing and elevated emotional friction."
            )
        elif energy_idx >= 75.0 and focus_idx >= 75.0:
            return (
                "Restored High-Flow Harmony",
                "Restful lifestyle recovery aligns with robust vocal acoustics, fluent pacing, and positive affect."
            )
        elif energy_idx >= 60.0 and focus_idx >= 60.0:
            return (
                "Steady Baseline Equilibrium",
                "All sensory streams are operating steadily within your typical personal baseline parameters."
            )
        else:
            return (
                "Moderate Cognitive Drag",
                "Mild fatigue and friction across linguistic and acoustic signals suggest pacing recalibration is beneficial."
            )

    @classmethod
    def generate_recommendations(
        cls,
        archetype: str,
        energy_idx: float,
        focus_idx: float,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate 2-3 actionable, restorative micro-recommendations.
        """
        recs = []
        if "Sleep-Deprived" in archetype or energy_idx < 50.0:
            recs.append("Prioritize an 8-hour sleep recovery window tonight and avoid intense screen-time after 9 PM.")
            recs.append("Protect your focus with 25-minute Pomodoro intervals rather than continuous 2-hour blocks.")
        elif "High-Arousal" in archetype:
            recs.append("Take a 5-minute diaphragmatic breathing or physical walk break to lower sympathetic nervous system tone.")
            recs.append("Delegate or postpone 1 non-urgent task to reduce immediate workload pressure.")
        elif "Masked Strain" in archetype:
            recs.append("Give yourself permission to acknowledge fatigue instead of powering through at 100% capacity.")
            recs.append("Schedule a 15-minute unplugged decompression window before your next major task.")
        elif "Restored" in archetype:
            recs.append("Leverage this high-flow window for your most demanding creative or strategic tasks.")
            recs.append("Maintain hydration and steady nutrition to preserve sustained afternoon stamina.")
        else:
            recs.append("Take a short 10-minute movement break to re-energize circulation and vocal projection.")
            recs.append("Keep tasks structured in single-threaded sprints to avoid divided attention.")

        return recs[:2]

    @classmethod
    def synthesize(cls, package: FrequencyInput) -> FusionData:
        """
        Execute full cross-modal fusion on a FrequencyInput evidence package.
        """
        raw_context = package.contextual_data.to_dict() if hasattr(package.contextual_data, "to_dict") else (package.contextual_data or {})
        raw_voice = package.voice_data.to_dict() if hasattr(package.voice_data, "to_dict") else (package.voice_data or {})
        raw_text = package.text_data.to_dict() if hasattr(package.text_data, "to_dict") else (package.text_data if isinstance(package.text_data, dict) else {})
        raw_baseline = package.baseline_data.to_dict() if hasattr(package.baseline_data, "to_dict") else (package.baseline_data or {})

        energy_idx = cls.compute_energy_index(raw_context, raw_voice, raw_text, raw_baseline)
        focus_idx = cls.compute_focus_index(raw_context, raw_voice, raw_text)
        align_score = cls.compute_alignment_score(raw_context, raw_voice, raw_text)

        archetype_title, archetype_desc = cls.classify_archetype(
            energy_idx, focus_idx, align_score, raw_context, raw_voice, raw_text
        )

        recs = cls.generate_recommendations(archetype_title, energy_idx, focus_idx, raw_context)

        matrix_summary = {
            "energy_index_pct": energy_idx,
            "cognitive_focus_pct": focus_idx,
            "sense_alignment_pct": align_score,
            "archetype": archetype_title,
            "primary_driver": "Sleep & Acoustic Pacing" if raw_context.get("sleep_hours", 7.5) < 6.0 else "Workload & Emotional Valence"
        }

        return FusionData(
            energy_index=energy_idx,
            focus_index=focus_idx,
            alignment_score=align_score,
            archetype=archetype_title,
            archetype_description=archetype_desc,
            recommendations=recs,
            matrix_summary=matrix_summary,
            confidence=0.92
        )
