from typing import Dict, Any, List, Optional, Union
from schema import FrequencyInput, TextSignalData, VoiceSignalData, ContextData, BaselineData, FusionData

class PromptBuilder:
    """
    Compiler that transforms structured FrequencyInput evidence packages 
    into intelligent, conversational LLM reasoning prompts.
    """

    SYSTEM_ROLE_PROMPT = (
        "You are FREQUENCY, an empathetic, highly intelligent Cross-Sense AI companion.\n\n"
        "YOUR CORE PURPOSE:\n"
        "You are having a real, meaningful conversation with the user. You synthesize their spoken/written words, "
        "their vocal tone/acoustics, lifestyle context (sleep, workload), and personal baselines to provide deep, "
        "supportive, and actionable insights.\n\n"
        "MANDATORY CONVERSATIONAL RULES:\n"
        "1. DUAL SYNTHESIS: ADDRESS THE USER'S ACTUAL TOPIC FIRST:\n"
        "   - ALWAYS respond directly and thoughtfully to the specific topic, problem, question, or story the user spoke or wrote about.\n"
        "   - Connect what they said with how they sounded (e.g. 'I hear how frustrated you are with the project deadline, and the hesitation in your voice highlights the mental strain you are under. Here is how we can tackle it...').\n"
        "   - NEVER give an answer that only discusses voice acoustics while ignoring the actual topic or question they shared!\n"
        "2. NO ROBOTIC META-OPENINGS:\n"
        "   - NEVER start with 'You've shared a voice note', 'I noticed from your acoustic cues', 'Based on your audio telemetry', or similar robotic phrases.\n"
        "   - Converse naturally, warmly, and directly as a human companion would.\n"
        "3. MODALITY INTEGRITY:\n"
        "   - If the user typed text, treat it purely as text (never invent vocal pitch or tone).\n"
        "   - If the user spoke audio, weave in vocal pacing, energy, and hesitation to enrich your response to their topic.\n"
        "4. COMPLETE & SUPPORTIVE:\n"
        "   - Conclude all thoughts and sentences cleanly. Offer 1-2 practical reflections or micro-restorative suggestions."
    )

    @classmethod
    def _format_dict_items(cls, data: Union[Dict[str, Any], Any], indent_level: int = 0) -> List[str]:
        """Recursively format dictionary or dataclass entries into clean markdown bullet points."""
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif not isinstance(data, dict):
            return [f"{'  ' * indent_level}- {data}"]

        lines = []
        prefix = "  " * indent_level + "- "
        sub_prefix = "  " * (indent_level + 1) + "- "

        for key, val in data.items():
            formatted_key = key.replace('_', ' ').title()
            if hasattr(val, "to_dict"):
                val = val.to_dict()

            if isinstance(val, dict):
                if val:
                    lines.append(f"{prefix}{formatted_key}:")
                    for sub_k, sub_v in val.items():
                        formatted_sub_k = sub_k.replace('_', ' ').title()
                        if isinstance(sub_v, list):
                            lines.append(f"{sub_prefix}{formatted_sub_k}: {', '.join(map(str, sub_v))}")
                        else:
                            lines.append(f"{sub_prefix}{formatted_sub_k}: {sub_v}")
            elif isinstance(val, list):
                if val:
                    lines.append(f"{prefix}{formatted_key}: {', '.join(map(str, val))}")
            elif val is not None and str(val).strip() != "":
                lines.append(f"{prefix}{formatted_key}: {val}")
        return lines

    @classmethod
    def build_system_prompt(cls) -> str:
        """Return the core system prompt enforcing natural conversation and evidence boundaries."""
        return cls.SYSTEM_ROLE_PROMPT

    @classmethod
    def build_evidence_prompt(cls, package: FrequencyInput) -> str:
        """
        Compile a FrequencyInput package into a clean evidence prompt for the LLM.
        """
        has_text_signals = bool(package.text_data)
        has_voice_data = bool(package.voice_data)
        has_context_data = bool(package.contextual_data)
        has_baseline_data = bool(package.baseline_data)
        has_fusion_data = bool(package.fusion_data)

        # Check if rich signals are present
        if isinstance(package.text_data, TextSignalData):
            has_rich_text = bool(package.text_data.sentiment or package.text_data.emotional_signals or package.text_data.detected_themes)
        elif isinstance(package.text_data, dict):
            has_rich_text = bool(package.text_data.get("sentiment") or package.text_data.get("emotional_signals") or package.text_data.get("detected_themes"))
        else:
            has_rich_text = False

        if isinstance(package.voice_data, VoiceSignalData):
            has_rich_voice = bool(package.voice_data.vocal_energy_rms is not None or package.voice_data.mean_pitch_hz is not None)
        elif isinstance(package.voice_data, dict):
            has_rich_voice = bool(package.voice_data.get("vocal_energy_rms") is not None or package.voice_data.get("mean_pitch_hz") is not None)
        else:
            has_rich_voice = False

        has_multimodal_evidence = (
            has_rich_text or has_rich_voice or has_context_data or has_baseline_data or has_fusion_data
        )

        if not has_multimodal_evidence:
            return package.user_prompt.strip()

        sections: List[str] = []

        # Prominent User Content Section
        if has_rich_voice:
            sections.append("## USER SPOKEN STATEMENT (ADDRESS THIS CONTENT & MEANING DIRECTLY):")
            sections.append(f"\"{package.user_prompt.strip()}\"")
            sections.append("\n*Instructions: Respond directly to what the user said above, while weaving in acoustic observations naturally.*")
        else:
            sections.append("## USER TYPED MESSAGE (ADDRESS THIS CONTENT DIRECTLY):")
            sections.append(f"\"{package.user_prompt.strip()}\"")
            sections.append("\n*(Note: User typed this message. No voice audio was captured. Do not comment on vocal tone or pitch.)*")

        sections.append("\n## MULTIMODAL CONTEXT & SENSORY SIGNALS (FOR YOUR INTERNAL REASONING)")
        
        # 1. Text Signals
        if has_text_signals:
            sections.append("### Extracted Text & Linguistic Signals:")
            sections.extend(cls._format_dict_items(package.text_data, indent_level=1))

        # 2. Voice Acoustics (Phase 3)
        if has_rich_voice:
            sections.append("### Extracted Voice Acoustics (Tone, Pacing, Energy):")
            sections.extend(cls._format_dict_items(package.voice_data, indent_level=1))

        # 3. User Context (Phase 4)
        if has_context_data:
            sections.append("### Lifestyle Context (Sleep, Workload, Mood):")
            sections.extend(cls._format_dict_items(package.contextual_data, indent_level=1))

        # 4. Baselines (Phase 5)
        if has_baseline_data:
            sections.append("### Personal Baselines & Deviations:")
            sections.extend(cls._format_dict_items(package.baseline_data, indent_level=1))

        # 5. Cross-Sense Fusion Matrix (Phase 6)
        if has_fusion_data:
            sections.append("### Cross-Sense Fusion Synthesis & Archetype:")
            sections.extend(cls._format_dict_items(package.fusion_data, indent_level=1))

        sections.append(
            "\n## INSTRUCTION\n"
            "1. Respond directly and meaningfully to what the user spoke/wrote about above.\n"
            "2. Weave in their vocal tone, sleep, and workload to explain *why* they might be feeling this way.\n"
            "3. Do NOT open with meta-phrases ('You shared a voice note', 'I noticed from your voice').\n"
            "4. Maintain a supportive, conversational dialogue."
        )

        return "\n".join(sections)

    @classmethod
    def build_messages(cls, package: FrequencyInput) -> List[Dict[str, str]]:
        """Compile chat messages array."""
        return [
            {"role": "system", "content": cls.build_system_prompt()},
            {"role": "user", "content": cls.build_evidence_prompt(package)}
        ]

    @classmethod
    def preview(cls, package: FrequencyInput) -> Dict[str, Any]:
        """Generate preview payload for inspector."""
        return {
            "system_prompt": cls.build_system_prompt(),
            "evidence_prompt": cls.build_evidence_prompt(package),
            "active_modalities": package.get_active_modalities(),
            "package_summary": package.to_dict()
        }
