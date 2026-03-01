"""
Multilingual Text Emotion Inference Module — v2.1

Provides emotion detection from text input using the trained MuRIL model.
Supports English, Hindi, and Hinglish (code-mixed) text.

v2.1 changes:
- 8-class classification (added 'suppressed')
- Softmax instead of sigmoid (single-label, not multi-label)
- Multi-turn context support via [TURN_X]...[CURRENT] format
"""

import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification

logger = logging.getLogger(__name__)

# ============================================================================
# v2.1 Emotion labels (8-class)
# ============================================================================
EMOTION_LABELS = [
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'suppressed'
]
NUM_LABELS = len(EMOTION_LABELS)

# Model version
MODEL_VERSION = "v2.1"

# Default model paths
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "text_emotion_v2" / "best"
FALLBACK_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "text_emotion" / "muril"


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class TextEmotionResult:
    """Basic emotion prediction result."""
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    text: str
    language: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'emotions': {k: round(v, 4) for k, v in self.emotions.items()},
            'dominant_emotion': self.dominant_emotion,
            'confidence': round(self.confidence, 4),
            'text': self.text,
            'language': self.language,
        }


@dataclass
class TextAnalysisResult:
    """
    Full analysis result — TextEmotionResult + incongruence detection.
    """
    # Model output
    emotions: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = 'neutral'
    confidence: float = 0.0

    # Incongruence detection (from suppressed class)
    incongruence: Dict = field(default_factory=lambda: {
        'detected': False, 'surface': '', 'implied': ''
    })

    # Language
    language: str = 'en'

    # Metadata
    context_turns_used: int = 0
    model_version: str = MODEL_VERSION
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'emotions': {k: round(v, 4) for k, v in self.emotions.items()},
            'dominant_emotion': self.dominant_emotion,
            'confidence': round(self.confidence, 4),
            'incongruence': self.incongruence,
            'language': self.language,
            'context_turns_used': self.context_turns_used,
            'model_version': self.model_version,
            'inference_time_ms': round(self.inference_time_ms, 1),
        }


# ============================================================================
# Classifier
# ============================================================================

class TextEmotionClassifier:
    """
    Multilingual text emotion classifier using fine-tuned MuRIL (v2.1).
    
    Supports:
    - English, Hindi (Devanagari), Hinglish (code-mixed) text
    - 8-class emotion classification including 'suppressed'
    - Multi-turn context via [TURN_X]...[CURRENT] format
    - Full TextAnalysisResult with integrated rule-based layers
    
    Usage:
        classifier = TextEmotionClassifier()
        result = classifier.predict("I am feeling happy today")
        full = classifier.analyze("Sab theek hai", turn_number=5)
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.3
    ):
        self.model_dir = Path(model_dir) if model_dir else self._find_model_dir()
        self.threshold = threshold
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._use_fallback = False
        self._fallback_pipeline = None



        logger.info(f"TextEmotionClassifier v2.1 initialized (device: {self.device})")
    
    def _find_model_dir(self) -> Path:
        """Find the best available model directory."""
        if DEFAULT_MODEL_DIR.exists():
            return DEFAULT_MODEL_DIR
        elif FALLBACK_MODEL_DIR.exists():
            logger.warning(f"Using fallback model: {FALLBACK_MODEL_DIR}")
            return FALLBACK_MODEL_DIR
        else:
            raise FileNotFoundError(
                f"No trained model found. Expected at:\n"
                f"  - {DEFAULT_MODEL_DIR}\n"
                f"  - {FALLBACK_MODEL_DIR}"
            )
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        logger.info(f"Loading model from {self.model_dir}...")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            config = AutoConfig.from_pretrained(str(self.model_dir))
            self._model = BertForSequenceClassification.from_pretrained(
                str(self.model_dir), config=config
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Model loaded (num_labels: {config.num_labels})")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            logger.info("Falling back to HuggingFace pretrained emotion model...")
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load a pretrained HuggingFace emotion model as fallback."""
        from transformers import pipeline
        FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"
        try:
            self._fallback_pipeline = pipeline(
                "text-classification", model=FALLBACK_MODEL, top_k=None,
                device=0 if self.device.type == 'cuda' else -1
            )
            self._use_fallback = True
            self._loaded = True
            self._fallback_label_map = {
                'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear',
                'joy': 'joy', 'sadness': 'sadness', 'surprise': 'surprise',
                'neutral': 'neutral'
            }
            logger.info("Fallback model loaded")
        except Exception as e2:
            logger.error(f"Fallback model also failed: {e2}")
            self._loaded = True
    
    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model
    
    @property
    def tokenizer(self):
        if not self._loaded:
            self.load()
        return self._tokenizer

    # ----------------------------------------------------------------
    # Core prediction
    # ----------------------------------------------------------------

    def predict(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> TextEmotionResult:
        """
        Predict emotions from text (model-only, no post-processing).
        
        Args:
            text: Input text (English, Hindi, or Hinglish)
            language: Optional detected language code
        
        Returns:
            TextEmotionResult with 8-class softmax probabilities
        """
        if not text or not text.strip():
            return TextEmotionResult(
                emotions={label: 0.0 for label in EMOTION_LABELS},
                dominant_emotion='neutral',
                confidence=1.0,
                text=text,
                language=language
            )
        
        if language is None:
            language = self._detect_language(text)
        
        if self._use_fallback and self._fallback_pipeline:
            return self._predict_with_fallback(text, language)
        
        # Tokenize
        encoding = self.tokenizer(
            text, truncation=True, max_length=128,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference — softmax for single-label (NOT sigmoid)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        
        emotions = {EMOTION_LABELS[i]: float(probs[i]) for i in range(NUM_LABELS)}
        dominant_idx = probs.argmax()
        dominant_emotion = EMOTION_LABELS[dominant_idx]
        confidence = float(probs[dominant_idx])

        # -- DISAMBIGUATION PATCH --
        try:
            from app.ml.fear_disambiguator import disambiguate_fear_suppressed
            disambiguation = disambiguate_fear_suppressed(
                full_input=text,
                predicted_label=dominant_emotion,
                predicted_confidence=confidence,
                all_probs=emotions,
            )
            if disambiguation.correction_applied:
                dominant_emotion = disambiguation.corrected_label
                confidence = emotions.get(dominant_emotion, confidence)
                logger.debug(
                    "Disambiguator fired: %s -> %s (rule: %s)",
                    disambiguation.original_label,
                    disambiguation.corrected_label,
                    disambiguation.rule_fired,
                )
        except ImportError:
            pass

        return TextEmotionResult(
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            text=text,
            language=language
        )

    # ----------------------------------------------------------------
    # Full analysis with rule-based layers
    # ----------------------------------------------------------------

    def analyze(
        self,
        text: str,
        language: Optional[str] = None,
        context_turns: int = 0,
    ) -> TextAnalysisResult:
        """
        Full analysis: model prediction + incongruence detection.

        Args:
            text: Input text (can include [TURN_X]...[CURRENT] markers)
            language: Optional detected language code
            context_turns: Number of context turns included in text

        Returns:
            TextAnalysisResult
        """
        start_time = time.perf_counter()

        # 1. Model prediction
        emotion_result = self.predict(text, language)

        # 2. Language
        if language is None:
            language = emotion_result.language or 'en'

        # 3. Incongruence (from suppressed class)
        incongruence = {'detected': False, 'surface': '', 'implied': ''}
        if emotion_result.dominant_emotion == 'suppressed':
            # Find the strongest negative emotion underneath
            negative_emotions = {
                k: v for k, v in emotion_result.emotions.items()
                if k in {'anger', 'disgust', 'fear', 'sadness'} and v > 0.05
            }
            if negative_emotions:
                implied = max(negative_emotions, key=negative_emotions.get)
            else:
                implied = 'distress'
            incongruence = {
                'detected': True,
                'surface': 'neutral',
                'implied': implied,
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TextAnalysisResult(
            emotions=emotion_result.emotions,
            dominant_emotion=emotion_result.dominant_emotion,
            confidence=emotion_result.confidence,
            incongruence=incongruence,
            language=language,
            context_turns_used=context_turns,
            model_version=MODEL_VERSION,
            inference_time_ms=elapsed_ms,
        )

    # ----------------------------------------------------------------
    # Legacy / helper methods
    # ----------------------------------------------------------------

    def _predict_with_fallback(self, text: str, language: str) -> TextEmotionResult:
        """Predict using fallback HuggingFace pipeline."""
        results = self._fallback_pipeline(text)[0] if self._fallback_pipeline else []
        emotions = {label: 0.0 for label in EMOTION_LABELS}
        for item in results:
            label = item['label'].lower()
            mapped = self._fallback_label_map.get(label, label)
            if mapped in emotions:
                emotions[mapped] = item['score']
        dominant_emotion = max(emotions, key=emotions.get)
        return TextEmotionResult(
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence=emotions[dominant_emotion],
            text=text,
            language=language
        )

    def _detect_language(self, text: str) -> str:
        """Detect language of text using character analysis."""
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if hindi_chars > len(text) * 0.3:
            return 'hi'
        return 'en'
    
    def get_active_emotions(
        self, result: TextEmotionResult, threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Get emotions above a confidence threshold."""
        thresh = threshold or self.threshold
        active = [(e, p) for e, p in result.emotions.items() if p >= thresh]
        return sorted(active, key=lambda x: x[1], reverse=True)


# ============================================================================
# Singleton + convenience functions
# ============================================================================

_classifier_instance: Optional[TextEmotionClassifier] = None


def get_text_emotion_classifier(
    model_dir: Optional[str] = None,
    force_reload: bool = False
) -> TextEmotionClassifier:
    """Get or create the singleton text emotion classifier."""
    global _classifier_instance
    if _classifier_instance is None or force_reload:
        _classifier_instance = TextEmotionClassifier(model_dir=model_dir)
    return _classifier_instance


def predict_text_emotion(
    text: str, language: Optional[str] = None
) -> TextEmotionResult:
    """Quick emotion prediction (model only)."""
    return get_text_emotion_classifier().predict(text, language)


def analyze_text_full(
    text: str,
    language: Optional[str] = None,
    context_turns: int = 0,
) -> TextAnalysisResult:
    """Full analysis: model + incongruence detection."""
    return get_text_emotion_classifier().analyze(
        text, language, context_turns
    )


def get_model_hash(model_dir: Optional[str] = None) -> str:
    """Compute SHA256 hash of the model weights for version locking."""
    d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    weights_file = d / "model.safetensors"
    if not weights_file.exists():
        return "NO_MODEL"
    sha = hashlib.sha256()
    with open(weights_file, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()[:16]


if __name__ == '__main__':
    print("Testing Text Emotion v2.1...")
    print(f"Model dir: {DEFAULT_MODEL_DIR}")
    print(f"Model hash: {get_model_hash()}")
    print()

    test_texts = [
        ("I am so happy today!", "en"),
        ("Mujhe bahut gussa aa raha hai yaar", "hinglish"),
        ("Sab theek hai, koi baat nahi", "hinglish"),
        ("I feel like nothing will ever change", "en"),
        ("This is a neutral statement.", "en"),
    ]

    classifier = TextEmotionClassifier()

    print("Results:")
    print("-" * 70)

    for text, lang in test_texts:
        result = classifier.analyze(text, language=lang)
        print(f"Text: {text}")
        print(f"  Emotion: {result.dominant_emotion} ({result.confidence:.2%})")
        print(f"  Language: {result.language}")
        top3 = sorted(result.emotions.items(), key=lambda x: -x[1])[:3]
        print(f"  Top3: {', '.join(f'{e}:{p:.2f}' for e,p in top3)}")
        print()
