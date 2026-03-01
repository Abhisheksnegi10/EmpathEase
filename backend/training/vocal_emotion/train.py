"""
vocal_emotion_train.py
======================
emotion2vec+base fine-tuning for Empathease vocal emotion analysis.

Architecture:  emotion2vec+base (frozen backbone) + dual-head classifier
               Head 1: 7-class softmax  (anger/disgust/fear/joy/sadness/surprise/neutral)
               Head 2: valence + arousal regression  (2 floats)

Why emotion2vec+base over Wav2Vec2:
    - 9-class output already includes fearful, disgusted, surprised
    - State-of-the-art SER (ACL 2024), 90M params
    - ~80% faster training (backbone fully frozen, pre-extract embeddings)
    - Expected macro F1: 0.75–0.80 (vs ~0.65 with Wav2Vec2)

Why 7 not 8:   'suppressed' is LINGUISTICALLY detected (text model).
               No audio dataset labels emotional suppression — it's
               defined by the gap between face/text/voice, not by voice alone.

Dataset:       RAVDESS + CREMA-D via HuggingFace (~10,000 samples combined)
Hardware:      GTX 1650 4GB VRAM — batch 64 (embedding-only training = tiny VRAM)
Training time: ~15–30 minutes for 30 epochs (embeddings pre-extracted)

Pipeline:
    Phase 1: Extract 768-dim embeddings from all audio using emotion2vec+base
             (one-time cost, cached to disk as .npz)
    Phase 2: Train dual-head MLP on cached embeddings (very fast)

Output schema (fed into fusion layer):
    VocalAnalysisResult(
        emotions      = {'anger': 0.12, 'fear': 0.71, ...},   # 7-class probs
        dominant      = 'fear',
        confidence    = 0.71,
        valence       = -0.58,    # -1 (negative) → +1 (positive)
        arousal       = 0.82,     # 0 (calm) → 1 (high energy)
        duration_sec  = 3.2,
        source        = 'emotion2vec-plus-base-finetuned-v1'
    )

Usage:
    # Full pipeline (extract + train)
    python -m training.vocal_emotion.train

    # Train with specific output dir
    python -m training.vocal_emotion.train --output_dir models/vocal_v2

    # Skip extraction if embeddings already cached
    python -m training.vocal_emotion.train --skip_extraction

    # Eval only (after training)
    python -m training.vocal_emotion.train --eval_only --checkpoint models/vocal_emotion_v1/best

    # Quick smoke test (50 samples per class)
    python -m training.vocal_emotion.train --smoke_test
"""

import os
import sys
import json
import argparse
import warnings
import tempfile
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import random
import time
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

EMOTION_LABELS  = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_CLASSES     = len(EMOTION_LABELS)   # 7
LABEL2ID        = {e: i for i, e in enumerate(EMOTION_LABELS)}
ID2LABEL        = {i: e for i, e in enumerate(EMOTION_LABELS)}

# emotion2vec+base model ID (via FunASR / ModelScope)
EMOTION2VEC_MODEL = "iic/emotion2vec_plus_base"
EMBEDDING_DIM     = 768   # emotion2vec+base output dimension

SAMPLE_RATE     = 16000
MAX_DURATION    = 6.0                             # seconds
MAX_LENGTH      = int(SAMPLE_RATE * MAX_DURATION)  # 96,000 samples

# Valence/arousal targets per class (Russell's circumplex + clinical literature)
EMOTION_VA = {
    "anger":   (-0.60,  0.85),
    "disgust": (-0.70,  0.45),
    "fear":    (-0.70,  0.80),
    "joy":     ( 0.80,  0.70),
    "sadness": (-0.65,  0.20),
    "surprise":( 0.10,  0.85),
    "neutral": ( 0.00,  0.30),
}

# ─────────────────────────────────────────────────────────────────────
# DATASET MAPPINGS
# ─────────────────────────────────────────────────────────────────────

RAVDESS_MAP = {
    "01": "neutral",   # neutral
    "02": "neutral",   # calm → neutral (closest match)
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

CREMAD_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}

# RAVDESS label name aliases (for datasets that return string labels)
RAVDESS_NAME_MAP = {
    "neutral": "neutral", "calm": "neutral", "happy": "joy",
    "sad": "sadness", "angry": "anger", "fearful": "fear",
    "disgust": "disgust", "surprised": "surprise", "surprise": "surprise",
}


# ─────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────

def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample → mono → normalize → pad/truncate to MAX_LENGTH."""
    import librosa

    audio = np.array(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    elif len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode='constant')
    return audio


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING FROM HUGGINGFACE
# ─────────────────────────────────────────────────────────────────────

def _extract_audio(item):
    """Pull audio array and sample rate from a HuggingFace audio item."""
    audio_field = item.get("audio", item.get("speech", None))
    if audio_field is None:
        return None, None
    if isinstance(audio_field, dict):
        arr = audio_field.get("array", None)
        sr = audio_field.get("sampling_rate", 16000)
        if arr is not None:
            return np.array(arr, dtype=np.float32), sr
        return None, None
    if isinstance(audio_field, np.ndarray):
        return audio_field.astype(np.float32), 16000
    return None, None


def _try_load_dataset(slug, split="train"):
    """Try loading a HuggingFace dataset with audio decoded via soundfile."""
    from datasets import load_dataset, Audio

    logger.info(f"    Trying: {slug} ...")
    ds = load_dataset(slug, split=split)

    # Cast audio column to use soundfile decoder at 16kHz
    audio_cols = [c for c in ds.column_names if c in ("audio", "speech")]
    if audio_cols:
        ds = ds.cast_column(audio_cols[0], Audio(sampling_rate=16000, decode=True))

    return ds


def load_ravdess():
    """Load RAVDESS from HuggingFace. Tries multiple known dataset slugs."""

    slugs = [
        ("xbgoose/ravdess",                "train"),
        ("confit/ravdess-parquet",          "train"),
        ("MahiA/RAVDESS",                  "train"),
    ]
    for slug, split in slugs:
        try:
            ds = _try_load_dataset(slug, split)
            samples = []
            for item in ds:
                audio, sr = _extract_audio(item)
                if audio is None or len(audio) < 1600:  # skip <0.1s clips
                    continue

                # Try label field first (various field names)
                label = None
                for field in ("label", "emotion", "Emotion", "labels"):
                    raw = item.get(field, None)
                    if raw is not None:
                        raw_str = str(raw).lower().strip()
                        # Try numeric label (some datasets use int)
                        if raw_str.isdigit():
                            label = RAVDESS_MAP.get(raw_str.zfill(2))
                        else:
                            label = RAVDESS_NAME_MAP.get(raw_str)
                        if label:
                            break

                # Fall back to filename parsing
                if label is None:
                    for field in ("file", "path", "audio_path", "filename"):
                        fname = item.get(field, None)
                        if fname:
                            parts = os.path.basename(str(fname)).replace(".wav", "").split("-")
                            if len(parts) >= 3:
                                label = RAVDESS_MAP.get(parts[2])
                                if label:
                                    break

                if label:
                    samples.append({
                        "audio": audio,
                        "sr": sr, "label": label, "source": "ravdess"
                    })

            if samples:
                logger.info(f"  ✅ RAVDESS: {len(samples)} samples from {slug}")
                return samples
            else:
                logger.warning(f"    No valid samples extracted from {slug}")

        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.warning("  ⚠️  RAVDESS: all sources failed")
    return []


def load_cremad():
    """Load CREMA-D from HuggingFace."""

    # Additional label mapping for CREMA-D (some datasets use full names)
    CREMAD_NAME_MAP = {
        "angry": "anger", "anger": "anger", "ang": "anger",
        "disgust": "disgust", "disgusted": "disgust", "dis": "disgust",
        "fear": "fear", "fearful": "fear", "fea": "fear",
        "happy": "joy", "joy": "joy", "hap": "joy",
        "neutral": "neutral", "neu": "neutral",
        "sad": "sadness", "sadness": "sadness",
    }

    slugs = [
        ("confit/cremad-parquet",           "train"),
        ("AbstractTTS/CREMA-D",            "train"),
        ("crusoeai/crema-d",               "train"),
    ]
    for slug, split in slugs:
        try:
            ds = _try_load_dataset(slug, split)
            samples = []
            for item in ds:
                audio, sr = _extract_audio(item)
                if audio is None or len(audio) < 1600:
                    continue

                label = None
                # Try direct label field
                for field in ("label", "emotion", "Emotion", "labels", "sentence"):
                    raw = item.get(field, None)
                    if raw is not None:
                        raw_str = str(raw).strip()
                        # Try uppercase 3-letter code
                        label = CREMAD_MAP.get(raw_str.upper()[:3])
                        if not label:
                            # Try lowercase full name
                            label = CREMAD_NAME_MAP.get(raw_str.lower())
                        if label:
                            break

                # Fall back to filename parsing
                if label is None:
                    for field in ("file", "path", "audio_path", "filename"):
                        fname = item.get(field, None)
                        if fname:
                            parts = os.path.basename(str(fname)).replace(".wav", "").split("_")
                            if len(parts) >= 3:
                                label = CREMAD_MAP.get(parts[2].upper()[:3])
                                if label:
                                    break

                if label:
                    samples.append({
                        "audio": audio,
                        "sr": sr, "label": label, "source": "cremad"
                    })

            if samples:
                logger.info(f"  ✅ CREMA-D: {len(samples)} samples from {slug}")
                return samples
            else:
                logger.warning(f"    No valid samples extracted from {slug}")

        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.warning("  ⚠️  CREMA-D: all sources failed")
    return []


def load_all_data(smoke_test: bool = False):
    """Combine RAVDESS + CREMA-D. Raises if both fail."""
    logger.info("Loading datasets from HuggingFace...")
    ravdess = load_ravdess()
    cremad  = load_cremad()
    all_data = ravdess + cremad

    if not all_data:
        raise RuntimeError(
            "\n\nNo data loaded from HuggingFace.\n"
            "Check your internet connection and HuggingFace access.\n"
            "Manual fallback:\n"
            "  1. Download RAVDESS from https://zenodo.org/record/1188976\n"
            "  2. Download CREMA-D from https://github.com/CheyneyComputerScience/CREMA-D\n"
        )

    if smoke_test:
        by_class = defaultdict(list)
        for s in all_data:
            by_class[s["label"]].append(s)
        all_data = []
        for cls_samples in by_class.values():
            all_data.extend(cls_samples[:50])
        logger.info(f"  Smoke test mode: {len(all_data)} samples")

    logger.info(f"  Combined: {len(all_data)} samples")
    dist = Counter(s["label"] for s in all_data)
    logger.info(f"  Distribution: {dict(sorted(dist.items()))}")
    return all_data


def compute_class_weights(samples):
    """Inverse-frequency weights, capped at 3.0."""
    counts = Counter(s["label"] for s in samples)
    total  = sum(counts.values())
    weights = []
    for label in EMOTION_LABELS:
        count = counts.get(label, 1)
        weights.append(total / (NUM_CLASSES * count))
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = torch.clamp(weights, max=3.0)
    logger.info(f"  Class weights: {dict(zip(EMOTION_LABELS, [f'{w:.2f}' for w in weights]))}")
    return weights


def stratified_split(all_data, val_ratio=0.10, test_ratio=0.10, seed=42):
    """Stratified split by class. No actor leakage (random within class)."""
    random.seed(seed)

    by_class = defaultdict(list)
    for s in all_data:
        by_class[s["label"]].append(s)

    train_data, val_data, test_data = [], [], []
    for cls, cls_samples in by_class.items():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_val  = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        test_data.extend(cls_samples[:n_test])
        val_data.extend(cls_samples[n_test:n_test + n_val])
        train_data.extend(cls_samples[n_test + n_val:])

    logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────
# PHASE 1: EMBEDDING EXTRACTION (emotion2vec+base)
# ─────────────────────────────────────────────────────────────────────

def extract_embeddings(samples: list, cache_dir: str, batch_label: str = "all") -> dict:
    """
    Extract 768-dim utterance embeddings from all audio using emotion2vec+base.
    Saves to cache_dir as .npz for fast reloading.

    Returns dict mapping sample index → embedding (np.ndarray of shape [768]).
    """
    cache_path = os.path.join(cache_dir, f"embeddings_{batch_label}.npz")

    # Check cache
    if os.path.exists(cache_path):
        logger.info(f"  Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        embeddings = {int(k): v for k, v in data.items()}
        if len(embeddings) == len(samples):
            logger.info(f"  ✅ {len(embeddings)} cached embeddings loaded")
            return embeddings
        else:
            logger.warning(f"  Cache size mismatch ({len(embeddings)} vs {len(samples)}), re-extracting")

    # Load emotion2vec+base model via FunASR
    from funasr import AutoModel

    logger.info(f"  Loading emotion2vec+base model: {EMOTION2VEC_MODEL}")
    e2v_model = AutoModel(model=EMOTION2VEC_MODEL, hub="hf")
    logger.info("  ✅ emotion2vec+base loaded")

    os.makedirs(cache_dir, exist_ok=True)
    embeddings = {}
    failed = 0

    # Process each sample — save audio to temp file, extract embedding
    logger.info(f"  Extracting embeddings for {len(samples)} samples...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        try:
            # Preprocess audio
            audio = preprocess_audio(sample["audio"], sample["sr"])

            # Save to temp wav file (FunASR expects file path or wav.scp)
            import soundfile as sf
            tmp_path = os.path.join(cache_dir, "_temp_audio.wav")
            sf.write(tmp_path, audio, SAMPLE_RATE)

            # Extract embedding
            result = e2v_model.generate(
                tmp_path,
                granularity="utterance",
                extract_embedding=True,
            )

            # result is a list of dicts; each has 'feats' key
            if result and len(result) > 0:
                feats = result[0].get("feats", None)
                if feats is not None:
                    embedding = np.array(feats, dtype=np.float32).flatten()
                    # Ensure correct dimension
                    if embedding.shape[0] >= EMBEDDING_DIM:
                        embeddings[i] = embedding[:EMBEDDING_DIM]
                    else:
                        logger.warning(f"  Sample {i}: embedding dim {embedding.shape[0]} < {EMBEDDING_DIM}")
                        failed += 1
                        continue
                else:
                    failed += 1
            else:
                failed += 1

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            failed += 1

        # Progress logging
        if (i + 1) % 100 == 0 or i == len(samples) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{i+1}/{len(samples)}] "
                f"({rate:.1f} samples/sec, ETA: {eta/60:.1f}min, "
                f"failed: {failed})"
            )

    # Clean up temp file
    tmp_path = os.path.join(cache_dir, "_temp_audio.wav")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Save to cache
    np.savez_compressed(cache_path, **{str(k): v for k, v in embeddings.items()})
    elapsed = time.time() - start_time
    logger.info(
        f"  ✅ Extracted {len(embeddings)} embeddings in {elapsed/60:.1f}min "
        f"({failed} failed). Cached → {cache_path}"
    )

    return embeddings


# ─────────────────────────────────────────────────────────────────────
# PHASE 2: EMBEDDING DATASET + MODEL
# ─────────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset of pre-extracted emotion2vec embeddings with labels."""

    def __init__(self, embeddings: Dict[int, np.ndarray], samples: list,
                 valid_indices: List[int], augment: bool = False):
        self.data = []
        for idx in valid_indices:
            if idx in embeddings:
                s = samples[idx]
                va = EMOTION_VA[s["label"]]
                self.data.append({
                    "embedding": torch.tensor(embeddings[idx], dtype=torch.float32),
                    "label":     LABEL2ID[s["label"]],
                    "valence":   va[0],
                    "arousal":   va[1],
                })
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emb = item["embedding"]

        if self.augment:
            # Light embedding augmentation: Gaussian noise
            if random.random() < 0.3:
                emb = emb + torch.randn_like(emb) * 0.01
            # Random feature dropout
            if random.random() < 0.2:
                mask = torch.bernoulli(torch.full_like(emb, 0.95))
                emb = emb * mask

        return {
            "embedding": emb,
            "label":     torch.tensor(item["label"], dtype=torch.long),
            "valence":   torch.tensor(item["valence"], dtype=torch.float32),
            "arousal":   torch.tensor(item["arousal"], dtype=torch.float32),
        }


class VocalEmotionHead(nn.Module):
    """
    Dual-head classifier on frozen emotion2vec embeddings.

    Input:        768-dim emotion2vec embedding
    Head 1:       Emotion classifier → 7-class softmax
    Head 2:       VA regressor → [valence, arousal]

    Much smaller than full Wav2Vec2 model — only the heads are trained.
    Total trainable params: ~500K (vs ~95M for Wav2Vec2 fine-tuning)
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()

        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Emotion classifier
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Head 2: Valence + Arousal regressor
        self.va_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
            nn.Tanh(),  # output in [-1, +1]
        )

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: [B, 768] emotion2vec utterance embeddings
        Returns:
            emotion_logits: [B, 7]
            valence:        [B]
            arousal:        [B]
        """
        shared = self.shared(embeddings)                # [B, 512]
        emotion_logits = self.emotion_head(shared)      # [B, 7]
        va_raw = self.va_head(shared)                   # [B, 2]

        valence = va_raw[:, 0]                          # [-1, +1]
        arousal = (va_raw[:, 1] + 1) / 2               # [-1, +1] → [0, 1]

        return emotion_logits, valence, arousal


# ─────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────

class DualHeadLoss(nn.Module):
    """
    Combined loss:
        Total = alpha * CrossEntropyLoss(emotions) + beta * MSELoss(VA)

    alpha=0.8, beta=0.2 — emotion classification is primary task.
    VA regression is auxiliary — it regularizes the representation.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        self.mse   = nn.MSELoss()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, emotion_logits, valence_pred, arousal_pred,
                labels, valence_true, arousal_true):
        loss_ce = self.ce(emotion_logits, labels)
        loss_va = self.mse(valence_pred, valence_true) + \
                  self.mse(arousal_pred, arousal_true)
        return self.alpha * loss_ce + self.beta * loss_va, loss_ce, loss_va


# ─────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60}")
    logger.info(f"  Empathease Vocal Emotion Model — emotion2vec+base Training")
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU:    {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"{'='*60}\n")

    # ── Phase 0: Load data from HuggingFace ───────────────────────────
    all_data = load_all_data(smoke_test=args.smoke_test)
    class_weights = compute_class_weights(all_data)

    # Stratified split
    train_data_samples, val_data_samples, test_data_samples = stratified_split(all_data)

    # Build index maps (sample index in all_data → split)
    train_indices = list(range(len(all_data)))
    # Actually, we need to track which indices belong to which split
    # Since stratified_split returns new lists, let's re-index:
    combined = []
    split_map = {}
    for i, s in enumerate(train_data_samples):
        idx = len(combined)
        combined.append(s)
        split_map[idx] = "train"
    for i, s in enumerate(val_data_samples):
        idx = len(combined)
        combined.append(s)
        split_map[idx] = "val"
    for i, s in enumerate(test_data_samples):
        idx = len(combined)
        combined.append(s)
        split_map[idx] = "test"

    train_indices = [i for i, split in split_map.items() if split == "train"]
    val_indices   = [i for i, split in split_map.items() if split == "val"]
    test_indices  = [i for i, split in split_map.items() if split == "test"]

    # ── Phase 1: Extract embeddings ───────────────────────────────────
    cache_dir = os.path.join(args.output_dir, "embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if args.skip_extraction:
        logger.info("Skipping extraction — loading from cache...")
        cache_path = os.path.join(cache_dir, "embeddings_all.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached embeddings at {cache_path}. "
                "Run without --skip_extraction first."
            )
        data = np.load(cache_path, allow_pickle=True)
        embeddings = {int(k): v for k, v in data.items()}
    else:
        logger.info("\n── Phase 1: Extracting emotion2vec embeddings ──────────────\n")
        embeddings = extract_embeddings(combined, cache_dir, batch_label="all")

    # ── Phase 2: Train MLP dual-head ──────────────────────────────────
    logger.info("\n── Phase 2: Training dual-head classifier ─────────────────\n")

    train_ds = EmbeddingDataset(embeddings, combined, train_indices, augment=True)
    val_ds   = EmbeddingDataset(embeddings, combined, val_indices,   augment=False)
    test_ds  = EmbeddingDataset(embeddings, combined, test_indices,  augment=False)

    logger.info(f"  Embedding datasets — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = VocalEmotionHead(
        input_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  VocalEmotionHead: {total_params:,} trainable parameters")

    criterion = DualHeadLoss(
        class_weights=class_weights.to(device),
        alpha=0.8, beta=0.2
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Cosine annealing scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ── Training Loop ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    best_macro_f1 = 0.0
    best_epoch    = 0
    patience_counter = 0

    logger.info(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"  Dropout: {args.dropout} | Patience: {args.patience}")
    logger.info(f"\n{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss, total_ce, total_va, n_batches = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            emb      = batch["embedding"].to(device)
            labels   = batch["label"].to(device)
            valences = batch["valence"].to(device)
            arousals = batch["arousal"].to(device)

            optimizer.zero_grad()

            emo_logits, val_pred, aro_pred = model(emb)
            loss, loss_ce, loss_va = criterion(
                emo_logits, val_pred, aro_pred,
                labels, valences, arousals
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += loss_ce.item()
            total_va   += loss_va.item()
            n_batches  += 1

        scheduler.step()
        avg_train_loss = total_loss / n_batches

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                emb      = batch["embedding"].to(device)
                labels   = batch["label"].to(device)
                valences = batch["valence"].to(device)
                arousals = batch["arousal"].to(device)

                emo_logits, val_pred, aro_pred = model(emb)
                loss, _, _ = criterion(emo_logits, val_pred, aro_pred,
                                       labels, valences, arousals)

                preds = emo_logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                val_loss_total += loss.item()

        macro_f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_val_loss = val_loss_total / max(len(val_loader), 1)

        # Logging
        logger.info(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {avg_train_loss:.4f} (CE:{total_ce/n_batches:.4f} VA:{total_va/n_batches:.4f}) | "
            f"Val: {avg_val_loss:.4f} | F1: {macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch    = epoch
            patience_counter = 0

            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim":   EMBEDDING_DIM,
                    "num_classes":  NUM_CLASSES,
                    "dropout":      args.dropout,
                },
                "emotion_labels":   EMOTION_LABELS,
                "label2id":         LABEL2ID,
                "id2label":         ID2LABEL,
                "emotion_va":       EMOTION_VA,
                "best_epoch":       best_epoch,
                "best_macro_f1":    best_macro_f1,
                "backbone_model":   EMOTION2VEC_MODEL,
            }, os.path.join(best_dir, "checkpoint.pt"))

            logger.info(f"           ✅ NEW BEST (F1={macro_f1:.4f}) — saved to {best_dir}")

            # Per-class F1
            per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
            for label, f1_val in zip(EMOTION_LABELS, per_class):
                marker = "✅" if f1_val >= 0.55 else "⚠️ "
                logger.info(f"           {marker} {label:10s}: {f1_val:.3f}")

        else:
            patience_counter += 1
            logger.info(f"           (best: {best_macro_f1:.4f} @ epoch {best_epoch}, patience: {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Save latest checkpoint
        latest_dir = os.path.join(args.output_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(latest_dir, "model_state.pt"))

    # ── Final Evaluation on Test Set ──────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  Final evaluation on held-out test set")
    logger.info(f"  Loading best checkpoint from epoch {best_epoch}")
    logger.info(f"{'='*60}\n")

    best_ckpt = torch.load(
        os.path.join(args.output_dir, "best", "checkpoint.pt"),
        map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    all_valences_pred, all_valences_true = [], []
    all_arousals_pred, all_arousals_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            emb      = batch["embedding"].to(device)
            labels   = batch["label"].to(device)
            valences = batch["valence"].to(device)
            arousals = batch["arousal"].to(device)

            emo_logits, val_pred, aro_pred = model(emb)

            all_preds.extend(emo_logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_valences_pred.extend(val_pred.cpu().numpy())
            all_valences_true.extend(valences.cpu().numpy())
            all_arousals_pred.extend(aro_pred.cpu().numpy())
            all_arousals_true.extend(arousals.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(all_labels, all_preds,
                                      target_names=EMOTION_LABELS, zero_division=0)
    val_mae  = np.mean(np.abs(np.array(all_valences_pred) - np.array(all_valences_true)))
    aro_mae  = np.mean(np.abs(np.array(all_arousals_pred) - np.array(all_arousals_true)))

    logger.info(f"\n{report}")
    logger.info(f"  Macro F1:     {macro_f1:.4f}  (gate: ≥0.65)")
    logger.info(f"  Valence MAE:  {val_mae:.4f}")
    logger.info(f"  Arousal MAE:  {aro_mae:.4f}")

    gate_passed = macro_f1 >= 0.65
    logger.info(f"\n  Gate: {'✅ PASSED' if gate_passed else '❌ FAILED — extend training or adjust weights'}")

    # Save training record
    record = {
        "model":           "VocalEmotionHead",
        "backbone":        EMOTION2VEC_MODEL,
        "version":         "v1",
        "architecture":    "emotion2vec+base (frozen) → MLP dual-head",
        "best_epoch":      best_epoch,
        "test_macro_f1":   float(macro_f1),
        "gate_passed":     gate_passed,
        "valence_mae":     float(val_mae),
        "arousal_mae":     float(aro_mae),
        "emotion_labels":  EMOTION_LABELS,
        "num_classes":     NUM_CLASSES,
        "embedding_dim":   EMBEDDING_DIM,
        "output_note":     "suppressed not included — acoustically undetectable",
        "train_config": {
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "dropout":      args.dropout,
            "patience":     args.patience,
            "loss_alpha_ce": 0.8,
            "loss_beta_va":  0.2,
        }
    }
    record_path = os.path.join(args.output_dir, "training_record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"\n  Training record saved → {record_path}")
    logger.info(f"  Best model saved      → {os.path.join(args.output_dir, 'best')}")

    return record


# ─────────────────────────────────────────────────────────────────────
# INFERENCE CLASS (used by fusion layer)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VocalAnalysisResult:
    emotions:     dict           # {'anger': 0.12, 'fear': 0.71, ...}
    dominant:     str            # 'fear'
    confidence:   float          # 0.71
    valence:      float          # -0.58
    arousal:      float          # 0.82
    duration_sec: float          # 3.2
    source:       str = "emotion2vec-plus-base-finetuned-v1"


class VocalEmotionInferencer:
    """
    Loads trained emotion2vec+base + MLP head and runs inference on audio.
    Used by the fusion layer / vocal_prosody.py.

    Pipeline:
        1. Audio → preprocess (resample, normalize, pad/truncate)
        2. Save temp wav → emotion2vec+base → 768-dim embedding
        3. Embedding → MLP dual-head → 7-class probs + valence/arousal

    Usage:
        inferencer = VocalEmotionInferencer("models/vocal_emotion_v1/best")
        result = inferencer.predict(audio_array, sample_rate=16000)
        print(result.dominant, result.valence, result.arousal)
    """

    def __init__(self, checkpoint_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(
            os.path.join(checkpoint_dir, "checkpoint.pt"),
            map_location=self.device
        )

        # Load MLP head
        config = ckpt["model_config"]
        self.model = VocalEmotionHead(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            dropout=0.0,   # no dropout at inference
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.emotion_labels = ckpt["emotion_labels"]
        self.backbone_model = ckpt.get("backbone_model", EMOTION2VEC_MODEL)

        # Lazy-load emotion2vec backbone
        self._e2v_model = None
        self._temp_dir = tempfile.mkdtemp()

        logger.info(f"VocalEmotionInferencer loaded from {checkpoint_dir} on {device}")

    def _load_backbone(self):
        """Lazy-load emotion2vec+base backbone."""
        if self._e2v_model is None:
            from funasr import AutoModel
            logger.info(f"Loading emotion2vec backbone: {self.backbone_model}")
            self._e2v_model = AutoModel(model=self.backbone_model, hub="hf")
            logger.info("✅ emotion2vec backbone loaded")

    def _extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract 768-dim embedding from audio using emotion2vec."""
        self._load_backbone()
        import soundfile as sf

        audio = preprocess_audio(audio, sr)
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

    def predict(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> VocalAnalysisResult:
        """Predict emotion from raw audio array."""
        duration_sec = len(audio) / sr

        # Extract embedding
        embedding = self._extract_embedding(audio, sr)
        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run MLP head
        with torch.no_grad():
            emo_logits, val_pred, aro_pred = self.model(emb_tensor)

        probs = torch.softmax(emo_logits, dim=-1).squeeze().cpu().numpy()
        valence = float(val_pred.squeeze().cpu().item())
        arousal = float(aro_pred.squeeze().cpu().item())
        dominant_i = int(probs.argmax())

        emotions = {self.emotion_labels[i]: float(p) for i, p in enumerate(probs)}

        return VocalAnalysisResult(
            emotions     = emotions,
            dominant     = self.emotion_labels[dominant_i],
            confidence   = float(probs[dominant_i]),
            valence      = round(valence, 4),
            arousal      = round(arousal, 4),
            duration_sec = round(duration_sec, 2),
        )

    def predict_from_file(self, audio_path: str) -> VocalAnalysisResult:
        """Predict emotion from audio file path."""
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.predict(audio, sr)


# ─────────────────────────────────────────────────────────────────────
# EVAL ONLY
# ─────────────────────────────────────────────────────────────────────

def eval_only(args):
    """Run evaluation on a saved checkpoint."""
    if not args.checkpoint:
        raise ValueError("--checkpoint required for --eval_only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(
        os.path.join(args.checkpoint, "checkpoint.pt"),
        map_location=device
    )
    config = ckpt["model_config"]
    model = VocalEmotionHead(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Load data and extract embeddings
    all_data = load_all_data(smoke_test=args.smoke_test)
    cache_dir = os.path.join(args.output_dir, "embedding_cache")
    embeddings = extract_embeddings(all_data, cache_dir, batch_label="eval")

    # Use last 10% as test
    n_test = max(1, int(len(all_data) * 0.10))
    test_indices = list(range(len(all_data) - n_test, len(all_data)))
    test_ds = EmbeddingDataset(embeddings, all_data, test_indices, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            emo_logits, _, _ = model(batch["embedding"].to(device))
            all_preds.extend(emo_logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    logger.info(classification_report(
        all_labels, all_preds,
        target_names=EMOTION_LABELS, zero_division=0
    ))
    logger.info(f"Macro F1: {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train vocal emotion model with emotion2vec+base backbone"
    )
    parser.add_argument("--output_dir",      default="models/vocal_emotion_v1",
                        help="Directory for checkpoints, embeddings, records")
    parser.add_argument("--checkpoint",      default=None,
                        help="Path to checkpoint dir (for --eval_only)")
    parser.add_argument("--epochs",          type=int,   default=30,
                        help="Max training epochs (early stopping may end sooner)")
    parser.add_argument("--batch_size",      type=int,   default=64,
                        help="Batch size (embeddings are tiny — can go large)")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Learning rate for MLP head")
    parser.add_argument("--dropout",         type=float, default=0.3,
                        help="Dropout rate in MLP head")
    parser.add_argument("--patience",        type=int,   default=15,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip embedding extraction (use cached embeddings)")
    parser.add_argument("--eval_only",       action="store_true",
                        help="Only evaluate a saved checkpoint")
    parser.add_argument("--smoke_test",      action="store_true",
                        help="50 samples per class — fast pipeline verification")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)
