"""
ML module for emotion detection and analysis.

Components:
- text_emotion: Multilingual text emotion classification (MuRIL-based)
- facial_affect: Facial expression recognition (EfficientNet-B0-based)
- vocal_prosody: Vocal emotion analysis (emotion2vec+base)
- fusion: Multimodal emotion fusion engine (STT-aware cascading)
"""

from app.ml.text_emotion import (
    TextEmotionClassifier,
    TextEmotionResult,
    get_text_emotion_classifier,
    predict_text_emotion,
    EMOTION_LABELS,
)

from app.ml.facial_affect import (
    FacialAffectInference,
    get_inference_engine as get_facial_affect_engine,
    predict_emotion as predict_facial_affect,
    EMOTION_LABELS as FACIAL_EMOTION_LABELS,
)

from app.ml.vocal_prosody import (
    VocalProsodyAnalyzer,
    VocalEmotionResult,
    get_vocal_analyzer,
    predict_vocal_emotion,
    EMOTION_LABELS as VOCAL_EMOTION_LABELS,
)

from app.ml.fusion import (
    EmotionFusionEngine,
    get_fusion_engine,
    fuse_emotions,
)

__all__ = [
    # Text emotion
    'TextEmotionClassifier',
    'TextEmotionResult',
    'get_text_emotion_classifier',
    'predict_text_emotion',
    'EMOTION_LABELS',
    # Facial affect
    'FacialAffectInference',
    'get_facial_affect_engine',
    'predict_facial_affect',
    'FACIAL_EMOTION_LABELS',
    # Vocal prosody
    'VocalProsodyAnalyzer',
    'VocalEmotionResult',
    'get_vocal_analyzer',
    'predict_vocal_emotion',
    'VOCAL_EMOTION_LABELS',
    # Fusion
    'EmotionFusionEngine',
    'get_fusion_engine',
    'fuse_emotions',
]
