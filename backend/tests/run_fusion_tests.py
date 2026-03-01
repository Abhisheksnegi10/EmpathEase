"""
Standalone fusion test runner — no pytest dependency.
Run:  python tests/run_fusion_tests.py
"""

import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.fusion import EmotionFusionEngine, _valence_zone
from app.schemas.emotion import FusedEmotionalState, UNIFIED_EMOTION_LABELS

passed = 0
failed = 0
errors = []


def check(name, condition, msg=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        errors.append(f"  FAIL {name}: {msg}")
        print(f"  FAIL {name}: {msg}")


# --- Mock helpers ---

@dataclass
class MockTextResult:
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    text: str = ""
    language: Optional[str] = "en"


@dataclass
class MockVocalResult:
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    valence: float = 0.0
    arousal: float = 0.5
    duration: float = 3.0


def mk_text(dominant, conf=0.7):
    e = {l: 0.02 for l in UNIFIED_EMOTION_LABELS}
    e[dominant] = conf
    rem = 1.0 - conf
    for l in UNIFIED_EMOTION_LABELS:
        if l != dominant:
            e[l] = rem / (len(UNIFIED_EMOTION_LABELS) - 1)
    return MockTextResult(emotions=e, dominant_emotion=dominant, confidence=conf)


def mk_face(dominant, conf=0.6):
    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    p = {l: 0.02 for l in labels}
    p[dominant] = conf
    rem = 1.0 - conf
    for l in labels:
        if l != dominant:
            p[l] = rem / (len(labels) - 1)
    return {"top_emotion": dominant, "confidence": conf, "all_probs": p}


def mk_vocal(dominant, conf=0.5, valence=0.0, arousal=0.5):
    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    e = {l: 0.02 for l in labels}
    e[dominant] = conf
    rem = 1.0 - conf
    for l in labels:
        if l != dominant:
            e[l] = rem / (len(labels) - 1)
    return MockVocalResult(emotions=e, dominant_emotion=dominant, confidence=conf,
                           valence=valence, arousal=arousal)


engine = EmotionFusionEngine()

# === 1. Label alignment ===
print("\n--- Label Alignment ---")
r = engine.fuse(mk_text("joy"), mk_face("joy"), mk_vocal("joy"))
check("8 labels in output", len(r.emotions) == 8)
check("all UNIFIED labels present", set(r.emotions.keys()) == set(UNIFIED_EMOTION_LABELS))

r2 = engine.fuse(face_result=mk_face("joy"))
check("face-only: suppressed=0", r2.emotions["suppressed"] == 0.0)

r3 = engine.fuse(text_result=mk_text("suppressed", 0.6))
check("text suppressed flows through", r3.emotions["suppressed"] > 0.0)
check("dominant is suppressed", r3.dominant_emotion == "suppressed")

# === 2. Weight arithmetic ===
print("\n--- Weight Arithmetic ---")
r = engine.fuse(mk_text("joy", 0.8), mk_face("sadness", 0.8), mk_vocal("anger", 0.8),
                stt_confidence=0.75)
check("medium: joy > sadness (text weighted)", r.emotions["joy"] > r.emotions["sadness"])

r = engine.fuse(mk_text("joy", 0.9), mk_face("sadness", 0.9), mk_vocal("sadness", 0.9),
                stt_confidence=0.90)
check("high tier: text dominates → joy", r.dominant_emotion == "joy")

r = engine.fuse(mk_text("joy", 0.99), mk_face("sadness", 0.8), mk_vocal("sadness", 0.8),
                stt_confidence=0.40)
check("low tier: text dropped → sadness", r.dominant_emotion == "sadness")

# === 3. STT cascade tiers ===
print("\n--- STT Cascade ---")
check("no STT → medium", engine._select_tier(None).text == 0.40)
check("0.90 → high", engine._select_tier(0.90).text == 0.60)
check("0.75 → medium", engine._select_tier(0.75).text == 0.40)
check("0.50 → low", engine._select_tier(0.50).text == 0.00)
check("0.85 boundary → high", engine._select_tier(0.85).text == 0.60)
check("0.65 boundary → medium", engine._select_tier(0.65).text == 0.40)

# === 4. Modality combos ===
print("\n--- Modality Combos ---")
r = engine.fuse(text_result=mk_text("sadness"))
check("text-only: modalities=['text']", r.modalities_used == ["text"])
check("text-only: dominant=sadness", r.dominant_emotion == "sadness")
check("text-only: facial=None", r.facial_emotion is None)

r = engine.fuse(face_result=mk_face("fear"))
check("face-only: dominant=fear", r.dominant_emotion == "fear")
check("face-only: modalities=['face']", r.modalities_used == ["face"])

r = engine.fuse(vocal_result=mk_vocal("anger"))
check("voice-only: dominant=anger", r.dominant_emotion == "anger")
check("voice-only: modalities=['voice']", r.modalities_used == ["voice"])

r = engine.fuse(text_result=mk_text("joy"), face_result=mk_face("joy"))
check("text+face: both in modalities", "text" in r.modalities_used and "face" in r.modalities_used)

r = engine.fuse(mk_text("joy"), mk_face("joy"), mk_vocal("joy"))
check("all three: 3 modalities", len(r.modalities_used) == 3)

r = engine.fuse()
check("no modalities: neutral default", r.dominant_emotion == "neutral" and r.confidence == 0.0)

# === 5. Incongruence (valence-zone) ===
print("\n--- Incongruence ---")
r = engine.fuse(text_result=mk_text("joy"), face_result=mk_face("sadness"))
check("joy vs sadness → incongruence=True", r.incongruence["detected"] is True)

r = engine.fuse(mk_text("fear"), mk_face("sadness"), mk_vocal("anger"))
check("fear+sadness+anger (all neg) → incongruence=False", r.incongruence["detected"] is False)

r = engine.fuse(text_result=mk_text("suppressed"), face_result=mk_face("neutral"))
check("suppressed(neg) vs neutral(neu) → incongruence=True", r.incongruence["detected"] is True)

r = engine.fuse(text_result=mk_text("anger"))
check("single modality → incongruence=False", r.incongruence["detected"] is False)

r = engine.fuse(text_result=mk_text("joy"), face_result=mk_face("surprise"))
check("joy+surprise (both pos) → incongruence=False", r.incongruence["detected"] is False)

# === 6. Valence/Arousal ===
print("\n--- Valence/Arousal ---")
r = engine.fuse(text_result=mk_text("joy", 0.9))
check("joy → positive valence", r.valence > 0.0)

r = engine.fuse(text_result=mk_text("anger", 0.9))
check("anger → negative valence", r.valence < 0.0)

r = engine.fuse(text_result=mk_text("neutral"), vocal_result=mk_vocal("neutral", valence=0.8, arousal=0.9))
check("vocal V/A blended → positive valence", r.valence > 0.0)

r = engine.fuse(text_result=mk_text("joy", 0.99))
check("valence clamped [-1,1]", -1.0 <= r.valence <= 1.0)
check("arousal clamped [0,1]", 0.0 <= r.arousal <= 1.0)

# === 7. Valence zone helper ===
print("\n--- Valence Zone Helper ---")
check("joy=pos", _valence_zone("joy") == "pos")
check("surprise=pos", _valence_zone("surprise") == "pos")
check("anger=neg", _valence_zone("anger") == "neg")
check("suppressed=neg", _valence_zone("suppressed") == "neg")
check("neutral=neu", _valence_zone("neutral") == "neu")
check("unknown→neu", _valence_zone("xyz") == "neu")

# === 8. Schema ===
print("\n--- FusedEmotionalState Schema ---")
s = FusedEmotionalState()
check("default dominant=neutral", s.dominant_emotion == "neutral")
check("default emotions has 8 keys", len(s.emotions) == 8)

d = FusedEmotionalState(dominant_emotion="joy", confidence=0.75).model_dump()
check("serialization works", d["dominant_emotion"] == "joy" and d["confidence"] == 0.75)

# === Summary ===
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(e)
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
