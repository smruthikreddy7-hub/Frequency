from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Union
import json

@dataclass
class TextSignalData:
    """
    Standardized Text Signal Schema for FREQUENCY (Phase 2).
    Represents linguistic, sentiment, affective, and cognitive load measurements.
    """
    raw_text: str = ""
    sentiment: Optional[str] = None           # "positive", "neutral", "negative"
    emotional_valence: Optional[float] = None # -1.0 to +1.0
    emotional_signals: Dict[str, float] = field(default_factory=dict)
    detected_themes: List[str] = field(default_factory=list)
    cognitive_load: Optional[str] = None      # "moderate", "elevated", "fragmented"
    cognitive_load_markers: List[str] = field(default_factory=list)
    word_count: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextSignalData":
        return cls(
            raw_text=data.get("raw_text", ""),
            sentiment=data.get("sentiment"),
            emotional_valence=data.get("emotional_valence"),
            emotional_signals=data.get("emotional_signals", {}) or {},
            detected_themes=data.get("detected_themes", []) or [],
            cognitive_load=data.get("cognitive_load"),
            cognitive_load_markers=data.get("cognitive_load_markers", []) or [],
            word_count=data.get("word_count", 0),
            confidence=data.get("confidence", 1.0)
        )

@dataclass
class VoiceSignalData:
    """
    Standardized Voice Acoustic Signal Schema for FREQUENCY (Phase 3).
    Represents pitch (F0), vocal energy, speech rate, and temporal pause metrics.
    """
    audio_duration_seconds: Optional[float] = None
    speech_rate_wpm: Optional[float] = None
    mean_pitch_hz: Optional[float] = None
    pitch_variation_stdev: Optional[float] = None
    vocal_energy_rms: Optional[float] = None
    pause_duration_ratio: Optional[float] = None
    transcript: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceSignalData":
        return cls(
            audio_duration_seconds=data.get("audio_duration_seconds"),
            speech_rate_wpm=data.get("speech_rate_wpm"),
            mean_pitch_hz=data.get("mean_pitch_hz"),
            pitch_variation_stdev=data.get("pitch_variation_stdev"),
            vocal_energy_rms=data.get("vocal_energy_rms"),
            pause_duration_ratio=data.get("pause_duration_ratio"),
            transcript=data.get("transcript"),
            confidence=data.get("confidence", 1.0)
        )

@dataclass
class ContextData:
    """
    User-reported daily contextual variables (Phase 4).
    """
    user_id: str = "default_user"
    date: str = ""
    sleep_hours: float = 7.5
    sleep_quality: str = "moderate"     # "restful", "moderate", "restless"
    workload: str = "normal"            # "light", "normal", "heavy", "burnout"
    activity_level: str = "moderate"    # "sedentary", "moderate", "active"
    mood_score: int = 7                 # 1 to 10 scale
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextData":
        return cls(
            user_id=data.get("user_id", "default_user"),
            date=data.get("date", ""),
            sleep_hours=float(data.get("sleep_hours", 7.5)),
            sleep_quality=data.get("sleep_quality", "moderate"),
            workload=data.get("workload", "normal"),
            activity_level=data.get("activity_level", "moderate"),
            mood_score=int(data.get("mood_score", 7)),
            notes=data.get("notes", "")
        )

@dataclass
class BaselineData:
    """
    Personal baseline historical distributions and percentage shifts (Phase 5).
    """
    user_id: str = "default_user"
    baseline_sleep_hours: Optional[float] = 7.5
    baseline_speech_rate_wpm: Optional[float] = 145.0
    baseline_vocal_energy_rms: Optional[float] = 0.038
    baseline_mood_score: Optional[float] = 7.2
    deviations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineData":
        return cls(
            user_id=data.get("user_id", "default_user"),
            baseline_sleep_hours=data.get("baseline_sleep_hours"),
            baseline_speech_rate_wpm=data.get("baseline_speech_rate_wpm"),
            baseline_vocal_energy_rms=data.get("baseline_vocal_energy_rms"),
            baseline_mood_score=data.get("baseline_mood_score"),
            deviations=data.get("deviations", {}) or {}
        )

@dataclass
class FusionData:
    """
    Cross-Sense Fusion Matrix & Composite Scores Schema (Phase 6 Capstone).
    """
    energy_index: float = 70.0               # 0 to 100%
    focus_index: float = 75.0                # 0 to 100%
    alignment_score: float = 85.0            # 0 to 100%
    archetype: str = "Steady Baseline Equilibrium"
    archetype_description: str = "Sensory streams are aligned and balanced."
    recommendations: List[str] = field(default_factory=list)
    matrix_summary: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionData":
        return cls(
            energy_index=float(data.get("energy_index", 70.0)),
            focus_index=float(data.get("focus_index", 75.0)),
            alignment_score=float(data.get("alignment_score", 85.0)),
            archetype=data.get("archetype", "Steady Baseline Equilibrium"),
            archetype_description=data.get("archetype_description", ""),
            recommendations=data.get("recommendations", []) or [],
            matrix_summary=data.get("matrix_summary", {}) or {},
            confidence=float(data.get("confidence", 0.9))
        )

@dataclass
class PackageMetadata:
    """Metadata container for request packaging."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = "session_default"
    schema_version: str = "1.4.0"

@dataclass
class FrequencyInput:
    """
    Unified Cross-Sense Evidence Package Schema for FREQUENCY.
    """
    user_prompt: str
    text_data: Optional[Union[TextSignalData, Dict[str, Any], str]] = None
    voice_data: Optional[Union[VoiceSignalData, Dict[str, Any]]] = None
    contextual_data: Union[ContextData, Dict[str, Any]] = field(default_factory=dict)
    baseline_data: Union[BaselineData, Dict[str, Any]] = field(default_factory=dict)
    fusion_data: Optional[Union[FusionData, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: asdict(PackageMetadata()))

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate integrity of the input package."""
        if not self.user_prompt or not self.user_prompt.strip():
            if self.voice_data:
                transcript = self.voice_data.transcript if isinstance(self.voice_data, VoiceSignalData) else self.voice_data.get("transcript")
                if transcript and transcript.strip():
                    self.user_prompt = transcript.strip()
                else:
                    return False, "user_prompt or voice transcript cannot be empty."
            else:
                return False, "user_prompt cannot be empty."
        
        return True, None

    def get_active_modalities(self) -> List[str]:
        """Detect which sense modalities are actively populated."""
        modalities = ["prompt"]
        if self.text_data:
            if isinstance(self.text_data, str) and self.text_data.strip():
                modalities.append("text")
            elif isinstance(self.text_data, dict) and bool(self.text_data):
                modalities.append("text")
            elif isinstance(self.text_data, TextSignalData) and bool(self.text_data.raw_text):
                modalities.append("text")
        if self.voice_data:
            if isinstance(self.voice_data, VoiceSignalData) and bool(self.voice_data.vocal_energy_rms is not None or self.voice_data.mean_pitch_hz is not None):
                modalities.append("voice")
            elif isinstance(self.voice_data, dict) and bool(self.voice_data):
                modalities.append("voice")
        if self.contextual_data:
            if isinstance(self.contextual_data, ContextData) or (isinstance(self.contextual_data, dict) and bool(self.contextual_data)):
                modalities.append("context")
        if self.baseline_data:
            if isinstance(self.baseline_data, BaselineData) or (isinstance(self.baseline_data, dict) and bool(self.baseline_data)):
                modalities.append("baseline")
        if self.fusion_data:
            if isinstance(self.fusion_data, FusionData) or (isinstance(self.fusion_data, dict) and bool(self.fusion_data)):
                modalities.append("fusion")
        return modalities

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataclass to dictionary."""
        d = asdict(self)
        if isinstance(self.text_data, TextSignalData):
            d["text_data"] = self.text_data.to_dict()
        if isinstance(self.voice_data, VoiceSignalData):
            d["voice_data"] = self.voice_data.to_dict()
        if isinstance(self.contextual_data, ContextData):
            d["contextual_data"] = self.contextual_data.to_dict()
        if isinstance(self.baseline_data, BaselineData):
            d["baseline_data"] = self.baseline_data.to_dict()
        if isinstance(self.fusion_data, FusionData):
            d["fusion_data"] = self.fusion_data.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrequencyInput":
        """Construct FrequencyInput from raw dictionary with safe defaults."""
        user_prompt = data.get("user_prompt", "")
        raw_text_data = data.get("text_data", None)
        
        if isinstance(raw_text_data, dict) and "sentiment" in raw_text_data:
            text_data = TextSignalData.from_dict(raw_text_data)
        else:
            text_data = raw_text_data

        raw_voice_data = data.get("voice_data", None)
        if isinstance(raw_voice_data, dict) and ("mean_pitch_hz" in raw_voice_data or "vocal_energy_rms" in raw_voice_data):
            voice_data = VoiceSignalData.from_dict(raw_voice_data)
        else:
            voice_data = raw_voice_data

        raw_context = data.get("contextual_data", {}) or {}
        if isinstance(raw_context, dict) and ("sleep_quality" in raw_context and "notes" in raw_context):
            contextual_data = ContextData.from_dict(raw_context)
        else:
            contextual_data = raw_context

        raw_baseline = data.get("baseline_data", {}) or {}
        if isinstance(raw_baseline, dict) and ("baseline_sleep_hours" in raw_baseline and "baseline_vocal_energy_rms" in raw_baseline):
            baseline_data = BaselineData.from_dict(raw_baseline)
        else:
            baseline_data = raw_baseline

        raw_fusion = data.get("fusion_data", None)
        if isinstance(raw_fusion, dict) and "energy_index" in raw_fusion:
            fusion_data = FusionData.from_dict(raw_fusion)
        else:
            fusion_data = raw_fusion

        metadata = data.get("metadata", {}) or asdict(PackageMetadata())

        return cls(
            user_prompt=user_prompt,
            text_data=text_data,
            voice_data=voice_data,
            contextual_data=contextual_data,
            baseline_data=baseline_data,
            fusion_data=fusion_data,
            metadata=metadata
        )

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Return the JSON schema representation for external contract validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "FrequencyInput",
            "description": "Unified Cross-Sense Evidence Package Schema with Fusion Engine & Multi-Modal Matrix Synthesis (Phase 6)",
            "type": "object",
            "required": ["user_prompt"],
            "properties": {
                "user_prompt": { "type": "string" },
                "text_data": { "type": ["object", "string", "null"] },
                "voice_data": { "type": ["object", "null"] },
                "contextual_data": { "type": "object" },
                "baseline_data": { "type": "object" },
                "fusion_data": { "type": ["object", "null"] },
                "metadata": { "type": "object" }
            }
        }
