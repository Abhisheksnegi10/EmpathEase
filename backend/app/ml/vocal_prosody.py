"""
Vocal Prosody Analysis Module.

Audio emotion recognition using emotion2vec+base (ACL 2024).
Detects emotional states from voice: anger, disgust, fear, joy, neutral, sadness, surprise.

Architecture:
    emotion2vec+base (frozen backbone, ~90M params)
    → 768-dim utterance embedding
    → Trained MLP dual-head (7-class classifier + VA regressor)

Uses lazy loading to avoid slow startup.

Usage:
    from app.ml import predict_vocal_emotion, get_vocal_analyzer

    # Single audio prediction
    result = predict_vocal_emotion(audio_array, sample_rate=16000)
    print(result.dominant_emotion, result.confidence)

    # Streaming chunks
    analyzer = get_vocal_analyzer()
    result = analyzer.analyze_chunk(audio_chunk, sample_rate=16000)
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

# emotion2vec+base model ID (via FunASR)
EMOTION2VEC_MODEL = "iic/emotion2vec_plus_base"
EMBEDDING_DIM     = 768

# Path to trained MLP checkpoint (relative to backend/)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "vocal_emotion", "best"
)

# Emotion labels (aligned with unified set)
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Audio config
SAMPLE_RATE    = 16000
MAX_DURATION   = 6.0
MAX_LENGTH     = int(SAMPLE_RATE * MAX_DURATION)
CHUNK_DURATION = 0.5


# ─────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VocalEmotionResult:
    """Result from vocal emotion prediction."""
    emotions: Dict[str, float]  # All emotion probabilities
    dominant_emotion: str       # Highest probability emotion
    confidence: float           # Confidence of dominant emotion
    valence: float              # Emotional valence (-1 to 1)
    arousal: float              # Emotional arousal (0 to 1)
    duration: float             # Audio duration in seconds


# ─────────────────────────────────────────────────────────────────────
# MLP HEAD (imported from training, or defined here for standalone use)
# ─────────────────────────────────────────────────────────────────────

def _build_head(input_dim: int = 768, num_classes: int = 7, dropout: float = 0.0):
    """Build the dual-head MLP (matching training architecture)."""
    import torch
    import torch.nn as nn

    class VocalEmotionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.emotion_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
            self.va_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 2),
                nn.Tanh(),
            )

        def forward(self, embeddings):
            shared = self.shared(embeddings)
            emotion_logits = self.emotion_head(shared)
            va_raw = self.va_head(shared)
            valence = va_raw[:, 0]
            arousal = (va_raw[:, 1] + 1) / 2
            return emotion_logits, valence, arousal

    return VocalEmotionHead()


# ─────────────────────────────────────────────────────────────────────
# ANALYZER
# ─────────────────────────────────────────────────────────────────────

class VocalProsodyAnalyzer:
    """
    Vocal prosody emotion analyzer using emotion2vec+base.

    Features:
    - Lazy loading (model loads on first prediction)
    - GPU acceleration if available
    - Streaming chunk support
    - Automatic resampling
    - Dual output: 7-class emotion + valence/arousal
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize analyzer.

        Args:
            checkpoint_dir: Path to trained MLP checkpoint directory
            device: 'cuda' or 'cpu', auto-detect if None
        """
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.device = device
        self._e2v_model = None
        self._head = None
        self._loaded = False
        self._temp_dir = tempfile.mkdtemp()
        self._fallback_mode = False

        logger.info(f"VocalProsodyAnalyzer initialized (lazy loading)")
        logger.info(f"Backbone: {EMOTION2VEC_MODEL}")
        logger.info(f"Checkpoint: {self.checkpoint_dir}")

    def _load_model(self) -> None:
        """Load emotion2vec backbone and MLP head."""
        if self._loaded:
            return

        import torch

        # Determine device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load emotion2vec+base backbone via FunASR
        try:
            from funasr import AutoModel

            logger.info(f"Loading emotion2vec backbone: {EMOTION2VEC_MODEL}")
            self._e2v_model = AutoModel(model=EMOTION2VEC_MODEL, hub="hf")
            logger.info("✅ emotion2vec backbone loaded")
        except Exception as e:
            logger.error(f"Failed to load emotion2vec: {e}")
            raise

        # Load trained MLP head
        ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading trained MLP head from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            config = ckpt["model_config"]
            self._head = _build_head(
                input_dim=config["input_dim"],
                num_classes=config["num_classes"],
                dropout=0.0,
            )
            self._head.load_state_dict(ckpt["model_state_dict"])
            self._head.to(self.device)
            self._head.eval()
            logger.info("✅ Trained MLP head loaded")
        else:
            # Fallback: use emotion2vec's built-in 9-class classifier
            logger.warning(
                f"No trained checkpoint at {ckpt_path}. "
                "Using emotion2vec built-in classifier (fallback mode)."
            )
            self._fallback_mode = True

        self._loaded = True
        logger.info("Vocal prosody model loaded successfully")

    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Preprocess audio for model input.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Original sample rate

        Returns:
            Preprocessed audio at 16kHz
        """
        # Convert to float32
        if audio.dtype != np.float32:
            if audio.dtype in [np.int16, np.int32]:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=SAMPLE_RATE
                )
            except ImportError:
                ratio = SAMPLE_RATE / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)

        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        # Truncate if too long
        max_samples = int(MAX_DURATION * SAMPLE_RATE)
        if len(audio) > max_samples:
            logger.warning(
                f"Audio truncated from {len(audio)/SAMPLE_RATE:.1f}s to {MAX_DURATION}s"
            )
            audio = audio[:max_samples]

        return audio

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract 768-dim embedding from preprocessed audio."""
        import soundfile as sf

        tmp_path = os.path.join(self._temp_dir, "inference_audio.wav")
        sf.write(tmp_path, audio, SAMPLE_RATE)

        result = self._e2v_model.generate(
            tmp_path,
            granularity="utterance",
            extract_embedding=True,
        )

        if result and len(result) > 0:
            feats = result[0].get("feats", None)
            if feats is not None:
                return np.array(feats, dtype=np.float32).flatten()[:EMBEDDING_DIM]

        raise RuntimeError("Failed to extract embedding from audio")

    def _fallback_predict(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Use emotion2vec's built-in 9-class classifier when no trained head exists.
        Maps 9 classes → our 7 classes.
        """
        import soundfile as sf

        tmp_path = os.path.join(self._temp_dir, "inference_audio.wav")
        sf.write(tmp_path, audio, SAMPLE_RATE)

        result = self._e2v_model.generate(
            tmp_path,
            granularity="utterance",
            extract_embedding=False,
        )

        # emotion2vec 9-class: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
        e2v_labels = ["anger", "disgust", "fear", "joy", "neutral", "other", "sadness", "surprise", "unknown"]
        our_map = {
            "anger": "anger", "disgust": "disgust", "fear": "fear",
            "joy": "joy", "neutral": "neutral", "sadness": "sadness",
            "surprise": "surprise", "other": "neutral", "unknown": "neutral",
        }

        emotions = {label: 0.0 for label in EMOTION_LABELS}

        if result and len(result) > 0:
            scores = result[0].get("scores", [])
            labels = result[0].get("labels", e2v_labels)

            for label, score in zip(labels, scores):
                label_lower = label.lower().replace("ed", "").replace("ful", "")
                # Map to our labels
                for e2v_lab, our_lab in our_map.items():
                    if e2v_lab in label_lower or label_lower in e2v_lab:
                        emotions[our_lab] += score
                        break

            # Normalize
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v / total for k, v in emotions.items()}

        return emotions

    def analyze(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: int = SAMPLE_RATE
    ) -> VocalEmotionResult:
        """
        Analyze audio for vocal emotion.

        Args:
            audio: Audio waveform (numpy array) or path to audio file
            sample_rate: Sample rate of input audio

        Returns:
            VocalEmotionResult with emotion predictions
        """
        self._load_model()

        import torch

        # Load from file if path
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))

        # Preprocess
        duration = len(audio) / sample_rate
        audio = self._preprocess_audio(audio, sample_rate)

        if self._fallback_mode:
            # Use emotion2vec's built-in classifier
            emotions = self._fallback_predict(audio)
        else:
            # Extract embedding → MLP head
            embedding = self._extract_embedding(audio)
            emb_tensor = torch.tensor(
                embedding, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emo_logits, val_pred, aro_pred = self._head(emb_tensor)

            probs = torch.softmax(emo_logits, dim=-1).squeeze().cpu().numpy()
            emotions = {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)}

        # Get dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        # Compute valence/arousal
        if not self._fallback_mode:
            valence = float(val_pred.squeeze().cpu().item())
            arousal = float(aro_pred.squeeze().cpu().item())
        else:
            valence = self._compute_valence(emotions)
            arousal = self._compute_arousal(emotions)

        return VocalEmotionResult(
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            duration=round(duration, 2),
        )

    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ) -> VocalEmotionResult:
        """
        Analyze a streaming audio chunk.

        For real-time analysis, send chunks of ~500ms–1s.
        Note: emotion2vec works best on longer utterances (2-6s).

        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate

        Returns:
            VocalEmotionResult for this chunk
        """
        return self.analyze(audio_chunk, sample_rate)

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            return audio, sr
        except ImportError:
            try:
                import librosa
                audio, sr = librosa.load(path, sr=None)
                return audio, sr
            except ImportError:
                raise ImportError("Install soundfile or librosa for audio loading")

    def _compute_valence(self, emotions: Dict[str, float]) -> float:
        """Compute emotional valence from probabilities (fallback mode)."""
        positive = emotions.get('joy', 0)
        negative = (
            emotions.get('anger', 0) +
            emotions.get('disgust', 0) +
            emotions.get('fear', 0) +
            emotions.get('sadness', 0)
        )
        return float(np.clip(positive - negative, -1, 1))

    def _compute_arousal(self, emotions: Dict[str, float]) -> float:
        """Compute emotional arousal from probabilities (fallback mode)."""
        high = (
            emotions.get('anger', 0) +
            emotions.get('fear', 0) +
            emotions.get('surprise', 0) +
            emotions.get('joy', 0) * 0.5
        )
        low = (
            emotions.get('sadness', 0) * 0.5 +
            emotions.get('neutral', 0)
        )
        return float(np.clip(0.5 + high - low, 0, 1))

    def get_labels(self) -> List[str]:
        """Get list of emotion labels."""
        return EMOTION_LABELS.copy()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# ─────────────────────────────────────────────────────────────────────
# SINGLETON + CONVENIENCE
# ─────────────────────────────────────────────────────────────────────

_analyzer: Optional[VocalProsodyAnalyzer] = None


def get_vocal_analyzer() -> VocalProsodyAnalyzer:
    """Get singleton vocal analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = VocalProsodyAnalyzer()
    return _analyzer


def predict_vocal_emotion(
    audio: Union[np.ndarray, str, Path],
    sample_rate: int = SAMPLE_RATE
) -> VocalEmotionResult:
    """
    Convenience function to predict vocal emotion.

    Args:
        audio: Audio waveform or path to audio file
        sample_rate: Sample rate of input audio

    Returns:
        VocalEmotionResult with emotion predictions
    """
    analyzer = get_vocal_analyzer()
    return analyzer.analyze(audio, sample_rate)
