"""
Unit tests for the multimodal emotion fusion engine.

Tests:
1. Label alignment (8-label text + 7-label face + 7-label voice → 8-label output)
2. Weight arithmetic (known inputs → expected weighted averages)
3. STT confidence cascade (high / medium / low tiers)
4. All modality combos (text-only, face-only, voice-only, pairs, all three)
5. Incongruence detection (valence-zone based, not label equality)
6. Valence / arousal computation
7. Edge cases (empty probs, zero confidence, no modalities)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import pytest

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.fusion import EmotionFusionEngine, _valence_zone
from app.schemas.emotion import FusedEmotionalState, UNIFIED_EMOTION_LABELS


# ============================================================================
# Test fixtures — mock modality results
# ============================================================================

@dataclass
class MockTextResult:
    """Mimics TextEmotionResult."""
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    text: str = ""
    language: Optional[str] = "en"


@dataclass
class MockVocalResult:
    """Mimics VocalEmotionResult."""
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    valence: float = 0.0
    arousal: float = 0.5
    duration: float = 3.0


def _make_text(dominant: str, confidence: float = 0.7) -> MockTextResult:
    """Make a text result with dominant emotion peaking."""
    emotions = {label: 0.02 for label in UNIFIED_EMOTION_LABELS}
    emotions[dominant] = confidence
    # Normalise remainder
    remainder = 1.0 - confidence
    other_labels = [l for l in UNIFIED_EMOTION_LABELS if l != dominant]
    for l in other_labels:
        emotions[l] = remainder / len(other_labels)
    return MockTextResult(
        emotions=emotions,
        dominant_emotion=dominant,
        confidence=confidence,
    )


def _make_face(dominant: str, confidence: float = 0.6) -> Dict:
    """Make a face result dict (7-label, uses 'top_emotion' key)."""
    face_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    probs = {l: 0.02 for l in face_labels}
    probs[dominant] = confidence
    remainder = 1.0 - confidence
    other = [l for l in face_labels if l != dominant]
    for l in other:
        probs[l] = remainder / len(other)
    return {
        "top_emotion": dominant,
        "confidence": confidence,
        "all_probs": probs,
    }


def _make_vocal(
    dominant: str, confidence: float = 0.5,
    valence: float = 0.0, arousal: float = 0.5,
) -> MockVocalResult:
    """Make a vocal result (7-label)."""
    vocal_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    emotions = {l: 0.02 for l in vocal_labels}
    emotions[dominant] = confidence
    remainder = 1.0 - confidence
    other = [l for l in vocal_labels if l != dominant]
    for l in other:
        emotions[l] = remainder / len(other)
    return MockVocalResult(
        emotions=emotions,
        dominant_emotion=dominant,
        confidence=confidence,
        valence=valence,
        arousal=arousal,
    )


# ============================================================================
# 1. Label alignment
# ============================================================================

class TestLabelAlignment:
    def test_fused_output_has_8_labels(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        face = _make_face("joy")
        vocal = _make_vocal("joy")

        result = engine.fuse(text, face, vocal)

        assert set(result.emotions.keys()) == set(UNIFIED_EMOTION_LABELS)
        assert len(result.emotions) == 8

    def test_suppressed_zero_for_face_only(self):
        engine = EmotionFusionEngine()
        face = _make_face("joy")

        result = engine.fuse(face_result=face)

        # suppressed should be 0 since face doesn't have it
        assert result.emotions["suppressed"] == 0.0

    def test_suppressed_nonzero_for_text(self):
        engine = EmotionFusionEngine()
        text = _make_text("suppressed", confidence=0.6)

        result = engine.fuse(text_result=text)

        assert result.emotions["suppressed"] > 0.0
        assert result.dominant_emotion == "suppressed"


# ============================================================================
# 2. Weight arithmetic
# ============================================================================

class TestWeights:
    def test_medium_tier_weights(self):
        """Medium tier: text=0.40, voice=0.30, face=0.30."""
        engine = EmotionFusionEngine()

        text = _make_text("joy", 0.8)
        face = _make_face("sadness", 0.8)
        vocal = _make_vocal("anger", 0.8)

        result = engine.fuse(text, face, vocal, stt_confidence=0.75)

        # joy should get most weight from text (0.40 * 0.8)
        assert result.emotions["joy"] > result.emotions["sadness"]

    def test_high_tier_text_dominates(self):
        """High STT → text=0.60."""
        engine = EmotionFusionEngine()

        text = _make_text("joy", 0.9)
        face = _make_face("sadness", 0.9)
        vocal = _make_vocal("sadness", 0.9)

        result = engine.fuse(text, face, vocal, stt_confidence=0.90)

        # Text should dominate even though face+voice say sadness
        assert result.dominant_emotion == "joy"

    def test_low_tier_text_dropped(self):
        """Low STT → text=0.00, text should have zero influence."""
        engine = EmotionFusionEngine()

        text = _make_text("joy", 0.99)
        face = _make_face("sadness", 0.8)
        vocal = _make_vocal("sadness", 0.8)

        result = engine.fuse(text, face, vocal, stt_confidence=0.40)

        # Text weight is 0.0 — dominant should be sadness
        assert result.dominant_emotion == "sadness"


# ============================================================================
# 3. STT cascade tiers
# ============================================================================

class TestSTTCascade:
    def test_no_stt_uses_medium(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(None)
        assert tier.text == 0.40

    def test_high_stt(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(0.90)
        assert tier.text == 0.60

    def test_medium_stt(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(0.75)
        assert tier.text == 0.40

    def test_low_stt(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(0.50)
        assert tier.text == 0.00

    def test_boundary_085(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(0.85)
        assert tier.text == 0.60  # >= 0.85 is high

    def test_boundary_065(self):
        engine = EmotionFusionEngine()
        tier = engine._select_tier(0.65)
        assert tier.text == 0.40  # >= 0.65 is medium


# ============================================================================
# 4. All modality combinations
# ============================================================================

class TestModalityCombos:
    def test_text_only(self):
        engine = EmotionFusionEngine()
        text = _make_text("sadness")

        result = engine.fuse(text_result=text)

        assert result.dominant_emotion == "sadness"
        assert result.modalities_used == ["text"]
        assert result.text_emotion == "sadness"
        assert result.facial_emotion is None
        assert result.vocal_emotion is None

    def test_face_only(self):
        engine = EmotionFusionEngine()
        face = _make_face("fear")

        result = engine.fuse(face_result=face)

        assert result.dominant_emotion == "fear"
        assert result.modalities_used == ["face"]
        assert result.facial_emotion == "fear"

    def test_voice_only(self):
        engine = EmotionFusionEngine()
        vocal = _make_vocal("anger", valence=-0.5, arousal=0.8)

        result = engine.fuse(vocal_result=vocal)

        assert result.dominant_emotion == "anger"
        assert result.modalities_used == ["voice"]
        assert result.vocal_emotion == "anger"

    def test_text_plus_face(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        face = _make_face("joy")

        result = engine.fuse(text_result=text, face_result=face)

        assert "text" in result.modalities_used
        assert "face" in result.modalities_used
        assert result.dominant_emotion == "joy"

    def test_text_plus_voice(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        vocal = _make_vocal("joy")

        result = engine.fuse(text_result=text, vocal_result=vocal)

        assert "text" in result.modalities_used
        assert "voice" in result.modalities_used

    def test_face_plus_voice(self):
        engine = EmotionFusionEngine()
        face = _make_face("neutral")
        vocal = _make_vocal("neutral")

        result = engine.fuse(face_result=face, vocal_result=vocal)

        assert "face" in result.modalities_used
        assert "voice" in result.modalities_used

    def test_all_three(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        face = _make_face("joy")
        vocal = _make_vocal("joy")

        result = engine.fuse(text, face, vocal)

        assert len(result.modalities_used) == 3
        assert result.dominant_emotion == "joy"

    def test_no_modalities_returns_neutral(self):
        engine = EmotionFusionEngine()
        result = engine.fuse()

        assert result.dominant_emotion == "neutral"
        assert result.confidence == 0.0


# ============================================================================
# 5. Incongruence detection (valence-zone based)
# ============================================================================

class TestIncongruence:
    def test_positive_vs_negative_fires(self):
        """text=joy (pos) + face=sadness (neg) → incongruence."""
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        face = _make_face("sadness")

        result = engine.fuse(text_result=text, face_result=face)

        assert result.incongruence["detected"] is True

    def test_two_negatives_no_incongruence(self):
        """text=fear (neg) + face=sadness (neg) + voice=anger (neg) → NO incongruence."""
        engine = EmotionFusionEngine()
        text = _make_text("fear")
        face = _make_face("sadness")
        vocal = _make_vocal("anger")

        result = engine.fuse(text, face, vocal)

        assert result.incongruence["detected"] is False

    def test_suppressed_vs_neutral_fires(self):
        """text=suppressed (neg) + face=neutral (neu) → incongruence."""
        engine = EmotionFusionEngine()
        text = _make_text("suppressed")
        face = _make_face("neutral")

        result = engine.fuse(text_result=text, face_result=face)

        assert result.incongruence["detected"] is True

    def test_single_modality_no_incongruence(self):
        """Only one modality → can't detect incongruence."""
        engine = EmotionFusionEngine()
        text = _make_text("anger")

        result = engine.fuse(text_result=text)

        assert result.incongruence["detected"] is False

    def test_all_positive_no_incongruence(self):
        """text=joy + face=surprise → both positive → no incongruence."""
        engine = EmotionFusionEngine()
        text = _make_text("joy")
        face = _make_face("surprise")

        result = engine.fuse(text_result=text, face_result=face)

        assert result.incongruence["detected"] is False


# ============================================================================
# 6. Valence / arousal
# ============================================================================

class TestValenceArousal:
    def test_joy_positive_valence(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy", 0.9)

        result = engine.fuse(text_result=text)

        assert result.valence > 0.0

    def test_anger_negative_valence(self):
        engine = EmotionFusionEngine()
        text = _make_text("anger", 0.9)

        result = engine.fuse(text_result=text)

        assert result.valence < 0.0

    def test_vocal_va_blended(self):
        """When vocal present, its V/A should influence the output."""
        engine = EmotionFusionEngine()
        text = _make_text("neutral")
        vocal = _make_vocal("neutral", valence=0.8, arousal=0.9)

        result = engine.fuse(text_result=text, vocal_result=vocal)

        # Vocal V/A = (0.8, 0.9) at 60% blend → should push valence positive
        assert result.valence > 0.0

    def test_valence_clamped(self):
        engine = EmotionFusionEngine()
        text = _make_text("joy", 0.99)

        result = engine.fuse(text_result=text)

        assert -1.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0


# ============================================================================
# 7. Valence zone helper
# ============================================================================

class TestValenceZone:
    def test_positive_zone(self):
        assert _valence_zone("joy") == "pos"
        assert _valence_zone("surprise") == "pos"

    def test_negative_zone(self):
        assert _valence_zone("anger") == "neg"
        assert _valence_zone("disgust") == "neg"
        assert _valence_zone("fear") == "neg"
        assert _valence_zone("sadness") == "neg"
        assert _valence_zone("suppressed") == "neg"

    def test_neutral_zone(self):
        assert _valence_zone("neutral") == "neu"

    def test_unknown_is_neutral(self):
        assert _valence_zone("unknown_label") == "neu"


# ============================================================================
# 8. FusedEmotionalState schema
# ============================================================================

class TestFusedSchema:
    def test_default_state(self):
        state = FusedEmotionalState()
        assert state.dominant_emotion == "neutral"
        assert state.confidence == 0.0
        assert len(state.emotions) == 8

    def test_serialization(self):
        state = FusedEmotionalState(
            dominant_emotion="joy",
            confidence=0.75,
            valence=0.6,
            arousal=0.5,
        )
        data = state.model_dump()
        assert data["dominant_emotion"] == "joy"
        assert data["confidence"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
