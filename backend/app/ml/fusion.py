"""
Multimodal Emotion Fusion Engine.

Combines text, facial, and vocal emotion predictions into a unified
FusedEmotionalState using STT-confidence-aware weight cascading.

Weight tiers (driven by Sarvam STT confidence):
    High   (≥0.85): text=0.60, vocal=0.30, face=0.10
    Medium (0.65–0.85): text=0.40, vocal=0.30, face=0.30
    Low    (<0.65): text=0.00, vocal=0.60, face=0.40

When a modality is absent, remaining weights renormalize to sum=1.0.

Incongruence detection uses *valence zones* (pos/neg/neu) rather than
label equality — two negative emotions disagreeing is NOT clinically
meaningful incongruence; positive vs negative IS.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

from app.schemas.emotion import FusedEmotionalState, UNIFIED_EMOTION_LABELS

logger = logging.getLogger(__name__)


# ============================================================================
# Valence zones (for incongruence detection)
# ============================================================================

_POSITIVE = {"joy", "surprise"}
_NEGATIVE = {"anger", "disgust", "fear", "sadness", "suppressed"}
_NEUTRAL  = {"neutral"}


def _valence_zone(emotion: str) -> str:
    """Map an emotion label to its valence zone."""
    if emotion in _POSITIVE:
        return "pos"
    if emotion in _NEGATIVE:
        return "neg"
    return "neu"


# ============================================================================
# Russell's circumplex mapping (emotion → valence/arousal)
# ============================================================================

_VAD_MAP: Dict[str, Tuple[float, float]] = {
    # (valence, arousal)
    "anger":      (-0.60, 0.80),
    "disgust":    (-0.65, 0.45),
    "fear":       (-0.70, 0.85),
    "joy":        ( 0.80, 0.65),
    "sadness":    (-0.70, 0.25),
    "surprise":   ( 0.20, 0.85),
    "neutral":    ( 0.00, 0.30),
    "suppressed": (-0.40, 0.35),
}


# ============================================================================
# Weight cascade tiers
# ============================================================================

class _WeightTier:
    """Weight preset for a given STT confidence tier."""

    def __init__(self, text: float, voice: float, face: float):
        self.text = text
        self.voice = voice
        self.face = face

    def as_dict(self) -> Dict[str, float]:
        return {"text": self.text, "voice": self.voice, "face": self.face}


# Defaults — overridable via Settings
DEFAULT_TIERS = {
    "high":   _WeightTier(text=0.60, voice=0.30, face=0.10),
    "medium": _WeightTier(text=0.40, voice=0.30, face=0.30),
    "low":    _WeightTier(text=0.00, voice=0.60, face=0.40),
}


# ============================================================================
# Fusion engine
# ============================================================================

class EmotionFusionEngine:
    """
    Three-modality emotion fusion with STT-aware weight cascading.

    Usage:
        engine = EmotionFusionEngine()
        fused = engine.fuse(
            text_result=text_result,       # TextEmotionResult or None
            face_result=face_result,       # dict from FacialAffectInference or None
            vocal_result=vocal_result,     # VocalEmotionResult or None
            stt_confidence=0.72,           # Sarvam STT confidence or None
        )
    """

    def __init__(
        self,
        tiers: Optional[Dict[str, _WeightTier]] = None,
        incongruence_threshold: float = 0.3,
    ):
        self.tiers = tiers or DEFAULT_TIERS
        self.incongruence_threshold = incongruence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        text_result=None,
        face_result: Optional[Dict] = None,
        vocal_result=None,
        stt_confidence: Optional[float] = None,
    ) -> FusedEmotionalState:
        """
        Fuse available modalities into a single emotional state.

        Args:
            text_result:    TextEmotionResult (or None)
            face_result:    dict with 'top_emotion', 'confidence', 'all_probs' (or None)
            vocal_result:   VocalEmotionResult (or None)
            stt_confidence: Sarvam STT transcript confidence (0–1, or None)

        Returns:
            FusedEmotionalState
        """
        start = time.perf_counter()

        # 1. Determine weight tier
        tier = self._select_tier(stt_confidence)

        # 2. Normalise per-modality emotion dicts to 8-label union
        text_probs = self._align_labels(text_result.emotions, "text") if text_result else None
        face_probs = self._align_labels(face_result.get("all_probs", {}), "face") if face_result else None
        vocal_probs = self._align_labels(vocal_result.emotions, "voice") if vocal_result else None

        # 3. Extract per-modality dominants (normalise face key)
        text_dominant = text_result.dominant_emotion if text_result else None
        face_dominant = face_result.get("top_emotion") if face_result else None
        vocal_dominant = vocal_result.dominant_emotion if vocal_result else None

        # 4. Build weight map for available modalities
        raw_weights: Dict[str, float] = {}
        probs_map: Dict[str, Dict[str, float]] = {}

        if text_probs is not None:
            raw_weights["text"] = tier.text
            probs_map["text"] = text_probs
        if face_probs is not None:
            raw_weights["face"] = tier.face
            probs_map["face"] = face_probs
        if vocal_probs is not None:
            raw_weights["voice"] = tier.voice
            probs_map["voice"] = vocal_probs

        if not probs_map:
            # No modalities at all — return neutral default
            return FusedEmotionalState()

        # 5. Renormalise weights to sum=1.0
        weights = self._renormalize(raw_weights)

        # 6. Weighted average of probability vectors
        fused_emotions = {label: 0.0 for label in UNIFIED_EMOTION_LABELS}
        for modality, w in weights.items():
            for label in UNIFIED_EMOTION_LABELS:
                fused_emotions[label] += w * probs_map[modality].get(label, 0.0)

        # 7. Dominant + confidence
        dominant_emotion = max(fused_emotions, key=fused_emotions.get)
        confidence = fused_emotions[dominant_emotion]

        # 8. Valence / arousal
        valence, arousal = self._compute_valence_arousal(
            fused_emotions, vocal_result
        )

        # 9. Incongruence (valence-zone based)
        incongruence = self._detect_incongruence(
            text_dominant, face_dominant, vocal_dominant
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        modalities_used = list(weights.keys())

        logger.info(
            "Fusion complete: %s → %s (%.2f) | tier=%s | modalities=%s | %.1fms",
            {m: d for m, d in [("text", text_dominant), ("face", face_dominant), ("voice", vocal_dominant)] if d},
            dominant_emotion,
            confidence,
            self._tier_name(stt_confidence),
            modalities_used,
            elapsed_ms,
        )

        return FusedEmotionalState(
            dominant_emotion=dominant_emotion,
            confidence=round(confidence, 4),
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            emotions={k: round(v, 4) for k, v in fused_emotions.items()},
            text_emotion=text_dominant,
            facial_emotion=face_dominant,
            vocal_emotion=vocal_dominant,
            incongruence=incongruence,
            modalities_used=modalities_used,
            stt_confidence=stt_confidence,
        )

    # ------------------------------------------------------------------
    # Weight tier selection
    # ------------------------------------------------------------------

    def _select_tier(self, stt_confidence: Optional[float]) -> _WeightTier:
        """Pick weight tier based on STT confidence."""
        if stt_confidence is None:
            # No STT → text-only or face-only session, use medium
            return self.tiers["medium"]
        if stt_confidence >= 0.85:
            return self.tiers["high"]
        if stt_confidence >= 0.65:
            return self.tiers["medium"]
        return self.tiers["low"]

    @staticmethod
    def _tier_name(stt_confidence: Optional[float]) -> str:
        if stt_confidence is None:
            return "medium(no-stt)"
        if stt_confidence >= 0.85:
            return "high"
        if stt_confidence >= 0.65:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Label alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_labels(
        emotions: Dict[str, float], source: str
    ) -> Dict[str, float]:
        """
        Align a modality's label set to the 8-label union.

        Face and voice lack 'suppressed' → gets 0.0.
        """
        aligned = {label: 0.0 for label in UNIFIED_EMOTION_LABELS}
        for label, prob in emotions.items():
            if label in aligned:
                aligned[label] = float(prob)
        return aligned

    # ------------------------------------------------------------------
    # Weight renormalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _renormalize(weights: Dict[str, float]) -> Dict[str, float]:
        """Renormalise weights to sum=1.0 for available modalities."""
        total = sum(weights.values())
        if total <= 0:
            # Equal split fallback
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Valence / arousal
    # ------------------------------------------------------------------

    def _compute_valence_arousal(
        self,
        fused_emotions: Dict[str, float],
        vocal_result=None,
    ) -> Tuple[float, float]:
        """
        Compute valence and arousal.

        Strategy:
        - If vocal modality provided and has V/A from its dual-head MLP,
          blend vocal V/A (60%) with circumplex-derived V/A (40%).
        - Otherwise, use pure circumplex mapping from fused emotion probs.
        """
        # Circumplex-derived V/A from fused probs
        circ_v = sum(
            prob * _VAD_MAP.get(label, (0.0, 0.3))[0]
            for label, prob in fused_emotions.items()
        )
        circ_a = sum(
            prob * _VAD_MAP.get(label, (0.0, 0.3))[1]
            for label, prob in fused_emotions.items()
        )

        if vocal_result is not None and hasattr(vocal_result, "valence"):
            # Blend: 60% vocal MLP, 40% circumplex
            valence = 0.6 * vocal_result.valence + 0.4 * circ_v
            arousal = 0.6 * vocal_result.arousal + 0.4 * circ_a
        else:
            valence = circ_v
            arousal = circ_a

        return (
            max(-1.0, min(1.0, valence)),
            max(0.0, min(1.0, arousal)),
        )

    # ------------------------------------------------------------------
    # Incongruence detection (valence-zone based)
    # ------------------------------------------------------------------

    def _detect_incongruence(
        self,
        text_dominant: Optional[str],
        face_dominant: Optional[str],
        vocal_dominant: Optional[str],
    ) -> Dict:
        """
        Detect clinically meaningful cross-modal incongruence.

        Fires when available modalities span different valence zones
        (positive / negative / neutral).  Two negative emotions
        disagreeing (e.g. fear vs sadness) is NOT incongruence.
        """
        dominants = {
            "text": text_dominant,
            "face": face_dominant,
            "voice": vocal_dominant,
        }
        # Only consider modalities that are present
        active = {m: d for m, d in dominants.items() if d is not None}

        if len(active) < 2:
            return {"detected": False, "details": ""}

        zones = {m: _valence_zone(d) for m, d in active.items()}
        unique_zones = set(zones.values())

        if len(unique_zones) > 1:
            zone_details = ", ".join(
                f"{m}={d}({zones[m]})" for m, d in active.items()
            )
            return {
                "detected": True,
                "details": f"Valence-zone mismatch: {zone_details}",
            }

        return {"detected": False, "details": ""}


# ============================================================================
# Singleton + convenience
# ============================================================================

_engine: Optional[EmotionFusionEngine] = None


def get_fusion_engine(
    incongruence_threshold: float = 0.3,
) -> EmotionFusionEngine:
    """Get or create the singleton fusion engine."""
    global _engine
    if _engine is None:
        _engine = EmotionFusionEngine(
            incongruence_threshold=incongruence_threshold,
        )
    return _engine


def fuse_emotions(
    text_result=None,
    face_result: Optional[Dict] = None,
    vocal_result=None,
    stt_confidence: Optional[float] = None,
) -> FusedEmotionalState:
    """
    Convenience function — fuse available modality results.

    Args:
        text_result:    TextEmotionResult or None
        face_result:    dict from FacialAffectInference.predict() or None
        vocal_result:   VocalEmotionResult or None
        stt_confidence: Sarvam STT transcript confidence (0–1)

    Returns:
        FusedEmotionalState
    """
    return get_fusion_engine().fuse(
        text_result=text_result,
        face_result=face_result,
        vocal_result=vocal_result,
        stt_confidence=stt_confidence,
    )
