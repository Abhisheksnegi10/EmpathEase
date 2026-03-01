"""
Emotion schemas — shared types for fused emotional state.

Used by:
- app.ml.fusion (produces FusedEmotionalState)
- app.memory.working (stores FusedEmotionalState in session turns)
- app.api.routes.emotion (multimodal endpoint request/response)
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Unified label set (8-class: text's superset)
# ============================================================================

UNIFIED_EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "neutral", "suppressed",
]


# ============================================================================
# Core fused state — used everywhere downstream
# ============================================================================

class FusedEmotionalState(BaseModel):
    """
    Fused emotional state from one or more modalities.

    This is the canonical emotion representation passed to memory,
    therapy engine, and API responses.
    """
    dominant_emotion: str = Field("neutral", description="Top fused emotion")
    confidence: float = Field(0.0, ge=0, le=1, description="Fused confidence")
    valence: float = Field(0.0, ge=-1, le=1, description="Emotional valence (-1 neg … +1 pos)")
    arousal: float = Field(0.5, ge=0, le=1, description="Emotional arousal (0 calm … 1 intense)")

    emotions: Dict[str, float] = Field(
        default_factory=lambda: {e: 0.0 for e in UNIFIED_EMOTION_LABELS},
        description="All 8-label fused probabilities",
    )

    # Per-modality dominants (None if modality wasn't provided)
    text_emotion: Optional[str] = None
    facial_emotion: Optional[str] = None
    vocal_emotion: Optional[str] = None

    # Cross-modal incongruence
    incongruence: Dict = Field(
        default_factory=lambda: {"detected": False, "details": ""},
        description="Valence-zone disagreement across modalities",
    )

    # Which modalities contributed
    modalities_used: List[str] = Field(default_factory=list)

    # STT confidence that drove weight selection (None if no STT)
    stt_confidence: Optional[float] = None

    # Crisis level (set by crisis service, not fusion)
    crisis_level: str = Field("none", description="none | watch | moderate | urgent")


# ============================================================================
# Multimodal API request / response
# ============================================================================

class MultimodalRequest(BaseModel):
    """Request for multimodal emotion analysis."""
    text: Optional[str] = Field(None, max_length=5000, description="Text input")
    image_base64: Optional[str] = Field(None, description="Base64-encoded face image (JPEG/PNG)")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio (WAV/MP3)")
    audio_sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    language: Optional[str] = Field(None, description="Language hint (auto-detected if omitted)")
    stt_confidence: Optional[float] = Field(
        None, ge=0, le=1,
        description="Sarvam STT transcript confidence — drives weight cascade",
    )


class MultimodalResponse(BaseModel):
    """Response for multimodal emotion analysis."""
    success: bool = True
    fused_state: FusedEmotionalState
    inference_time_ms: float = Field(0.0, description="Total fusion latency")


# ============================================================================
# Working memory helpers (re-exported for app.memory.working compatibility)
# ============================================================================

class ConversationTurn(BaseModel):
    """Single turn in a conversation (used by working memory)."""
    turn_id: int
    timestamp: str
    user_input: str
    emotional_state: FusedEmotionalState
    system_response: str


class WorkingMemory(BaseModel):
    """Working memory state stored in Redis."""
    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    context_entities: List[str] = Field(default_factory=list)
    current_topic: Optional[str] = None
    therapeutic_approach: str = "person_centered"
