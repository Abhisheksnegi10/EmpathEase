"""
Multilingual Dataset Loader for Text Emotion Classification v2.1.

Supports:
- GoEmotions (English) — 43K samples, 28-class → 8-class mapping
- Bhaav (Hindi) — 16K samples, Hindi comprehension
- Hinglish Therapy CSV — 12K samples, all 8 classes, multi-turn context

Uses unified 8-class emotion mapping with suppressed class.
Implements stratified train/val/test split.
"""

import os
import csv
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer

# ============================================================================
# 8-class unified labels (v2.1)
# ============================================================================

EMOTION_LABELS = [
    'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'suppressed'
]
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

# ============================================================================
# Label Mappings
# ============================================================================

# Hindi/Bhaav → 8-class (Bhaav has 5 labels — no fear/disgust/suppressed signal)
HINDI_EMOTION_MAP = {
    # Bhaav labels (English)
    'anger': 'anger',
    'joy': 'joy',
    'happy': 'joy',
    'happiness': 'joy',
    'sad': 'sadness',
    'sadness': 'sadness',
    'suspense': 'fear',       # Map suspense → fear (similar arousal)
    'neutral': 'neutral',
    # Hindi labels (Devanagari)
    'गुस्सा': 'anger',
    'क्रोध': 'anger',
    'खुशी': 'joy',
    'आनंद': 'joy',
    'दुख': 'sadness',
    'उदासी': 'sadness',
    'डर': 'fear',
    'भय': 'fear',
    'सस्पेंस': 'fear',
    'घृणा': 'disgust',
    'आश्चर्य': 'surprise',
    'तटस्थ': 'neutral',
}

# GoEmotions 28-class label list
GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# GoEmotions single-label → 8-class mapping (used when only ONE label present)
GOEMOTION_TO_UNIFIED = {
    'admiration': 'joy', 'amusement': 'joy', 'anger': 'anger',
    'annoyance': 'anger', 'approval': 'joy', 'caring': 'joy',
    'confusion': 'neutral', 'curiosity': 'neutral', 'desire': 'joy',
    'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'disgust', 'excitement': 'joy', 'fear': 'fear',
    'gratitude': 'joy', 'grief': 'sadness', 'joy': 'joy',
    'love': 'joy', 'nervousness': 'fear', 'optimism': 'joy',
    'pride': 'joy', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'sadness', 'sadness': 'sadness', 'surprise': 'surprise',
    'neutral': 'neutral'
}

# Negative GoEmotions labels (for suppressed co-occurrence detection)
NEGATIVE_GOEMOTION_INDICES = {
    2,   # anger
    3,   # annoyance
    10,  # disapproval
    11,  # disgust
    12,  # embarrassment
    14,  # fear
    16,  # grief
    19,  # nervousness
    24,  # remorse
    25,  # sadness
}

NEUTRAL_IDX = 27  # neutral in GoEmotions


# ============================================================================
# Datasets
# ============================================================================

class GoEmotionsDataset(Dataset):
    """
    GoEmotions dataset with 8-class mapping.
    Handles suppressed via co-occurrence rule:
      neutral + negative emotion → suppressed
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = 'google/muril-base-cased',
        max_length: int = 128,
        split: str = 'train'
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = []
        self.labels = []

        split_file = {'train': 'train.tsv', 'validation': 'dev.tsv', 'test': 'test.tsv'}
        filepath = Path(data_path) / split_file.get(split, f'{split}.tsv')

        if not filepath.exists():
            raise FileNotFoundError(f"GoEmotions file not found: {filepath}")

        print(f"Loading GoEmotions {split} from {filepath}...")
        self._load_tsv(filepath)
        print(f"  Loaded {len(self.texts)} samples")

    def _load_tsv(self, filepath: Path):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue
                text = row[0].strip()
                label_str = row[1] if len(row) > 1 else ''

                if not text or not label_str:
                    continue

                try:
                    label_indices = [int(x) for x in label_str.split(',') if x.strip()]
                except ValueError:
                    label_indices = [NEUTRAL_IDX]

                # Check for suppressed: neutral + any negative co-occurring
                has_neutral = NEUTRAL_IDX in label_indices
                has_negative = any(i in NEGATIVE_GOEMOTION_INDICES for i in label_indices)

                if has_neutral and has_negative and len(label_indices) >= 2:
                    # Suppressed: surface neutral masking underlying negative
                    unified_label = EMOTION_TO_IDX['suppressed']
                else:
                    # Standard single-class mapping (use first label's mapping)
                    unified_labels = set()
                    for idx in label_indices:
                        if idx < len(GOEMOTION_LABELS):
                            original = GOEMOTION_LABELS[idx]
                            unified = GOEMOTION_TO_UNIFIED.get(original, 'neutral')
                            unified_labels.add(EMOTION_TO_IDX[unified])

                    # Take the most specific (non-neutral) label if multiple
                    if len(unified_labels) > 1:
                        unified_labels.discard(EMOTION_TO_IDX['neutral'])
                    unified_label = min(unified_labels) if unified_labels else EMOTION_TO_IDX['neutral']

                self.texts.append(text)
                self.labels.append(unified_label)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class BhaavDataset(Dataset):
    """
    Bhaav Hindi dataset. Provides Hindi comprehension for 5 classes.
    Does NOT contribute to fear/disgust/suppressed.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = 'google/muril-base-cased',
        max_length: int = 128,
        split: str = 'train'
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = []
        self.labels = []

        filepath = Path(data_path) / f'{split}.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Bhaav file not found: {filepath}")

        print(f"Loading Bhaav {split} from {filepath}...")
        self._load_csv(filepath)
        print(f"  Loaded {len(self.texts)} samples")

    def _load_csv(self, filepath: Path):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', row.get('sentence', '')).strip()
                label = row.get('label', row.get('emotion', '')).strip().lower()

                if text and label:
                    unified_label = HINDI_EMOTION_MAP.get(label)
                    if unified_label and unified_label in EMOTION_TO_IDX:
                        self.texts.append(text)
                        self.labels.append(EMOTION_TO_IDX[unified_label])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class HinglishTherapyDataset(Dataset):
    """
    Hinglish Therapy CSV dataset — primary source for all 8 classes.
    Handles multi-turn [TURN_X]...[CURRENT] format natively.
    Columns: full_input, emotion, language, has_stt_noise, context_turns, source
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = 'google/muril-base-cased',
        max_length: int = 128,
        indices: Optional[List[int]] = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = []
        self.labels = []

        import pandas as pd
        df = pd.read_csv(csv_path)

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        print(f"Loading Hinglish Therapy: {len(df)} samples...")

        for _, row in df.iterrows():
            text = str(row['full_input']).strip()
            emotion = str(row['emotion']).strip().lower()

            if text and emotion in EMOTION_TO_IDX:
                # Pass full [TURN_X]...[CURRENT] string to tokenizer unchanged
                self.texts.append(text)
                self.labels.append(EMOTION_TO_IDX[emotion])

        print(f"  Loaded {len(self.texts)} samples ({len(set(self.labels))} classes)")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# Stratified dataloaders
# ============================================================================

def stratified_split(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Stratified train/val/test split that ensures all classes appear in all splits.
    Returns (train_indices, val_indices, test_indices).
    """
    from collections import defaultdict
    random.seed(seed)

    # Group indices by label
    label_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        label = dataset.labels[i]
        label_to_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def get_dataloaders(
    goemotions_dir: str,
    bhaav_dir: Optional[str] = None,
    hinglish_csv: Optional[str] = None,
    batch_size: int = 16,
    tokenizer_name: str = 'google/muril-base-cased',
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get combined multilingual dataloaders with stratified split.

    Args:
        goemotions_dir: Path to GoEmotions TSV directory
        bhaav_dir: Path to Bhaav CSV directory (optional)
        hinglish_csv: Path to Hinglish therapy CSV (optional)
        batch_size: Batch size (16 for GTX 1650)
        tokenizer_name: HuggingFace tokenizer
        max_length: Max token length
        num_workers: Data loading workers
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # --- GoEmotions (has its own train/dev/test split) ---
    for split, target in [('train', train_datasets), ('validation', val_datasets), ('test', test_datasets)]:
        try:
            ds = GoEmotionsDataset(goemotions_dir, tokenizer_name, max_length, split)
            target.append(ds)
        except FileNotFoundError as e:
            print(f"  GoEmotions {split} not found: {e}")

    # --- Bhaav (split manually: train → train, test → val) ---
    if bhaav_dir and Path(bhaav_dir).exists():
        for split, target in [('train', train_datasets), ('test', val_datasets)]:
            try:
                ds = BhaavDataset(bhaav_dir, tokenizer_name, max_length, split)
                target.append(ds)
            except FileNotFoundError as e:
                print(f"  Bhaav {split} not found: {e}")

    # --- Hinglish Therapy CSV (stratified split) ---
    if hinglish_csv and Path(hinglish_csv).exists():
        import pandas as pd
        df = pd.read_csv(hinglish_csv)
        n = len(df)

        # Use sklearn for proper stratified split
        try:
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(
                df, test_size=0.2, stratify=df['emotion'], random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42
            )
            train_indices = train_df.index.tolist()
            val_indices = val_df.index.tolist()
            test_indices = test_df.index.tolist()
        except ImportError:
            # Fallback: manual stratified split
            print("  sklearn not available, using manual stratified split")
            full_ds_temp = HinglishTherapyDataset(hinglish_csv, tokenizer_name, max_length)
            train_indices, val_indices, test_indices = stratified_split(full_ds_temp)

        train_datasets.append(HinglishTherapyDataset(hinglish_csv, tokenizer_name, max_length, train_indices))
        val_datasets.append(HinglishTherapyDataset(hinglish_csv, tokenizer_name, max_length, val_indices))
        test_datasets.append(HinglishTherapyDataset(hinglish_csv, tokenizer_name, max_length, test_indices))

    # --- Combine ---
    train_combined = ConcatDataset(train_datasets) if train_datasets else None
    val_combined = ConcatDataset(val_datasets) if val_datasets else None
    test_combined = ConcatDataset(test_datasets) if test_datasets else None

    print(f"\nCombined datasets:")
    print(f"  Train: {len(train_combined) if train_combined else 0}")
    print(f"  Val:   {len(val_combined) if val_combined else 0}")
    print(f"  Test:  {len(test_combined) if test_combined else 0}")
    print(f"  Labels: {NUM_LABELS} classes: {EMOTION_LABELS}")

    # --- Verify all 8 classes present ---
    if train_combined:
        seen = set()
        for ds in train_datasets:
            if hasattr(ds, 'labels'):
                seen.update(set(ds.labels))
        missing = set(range(NUM_LABELS)) - seen
        if missing:
            missing_names = [EMOTION_LABELS[i] for i in missing]
            print(f"  [WARN] Classes missing from train: {missing_names}")
        else:
            print(f"  [OK] All 8 classes present in training set")

    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_combined, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if val_combined else None
    test_loader = DataLoader(
        test_combined, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if test_combined else None

    return train_loader, val_loader, test_loader


# ============================================================================
# CLI test
# ============================================================================

if __name__ == '__main__':
    import sys

    print("Testing v2.1 Dataset Loader (8-class)...")
    print(f"Labels: {EMOTION_LABELS}")
    print(f"Num labels: {NUM_LABELS}")

    goemotions_dir = sys.argv[1] if len(sys.argv) > 1 else 'd:/EmpathEase v1/data/goemotions'
    bhaav_dir = sys.argv[2] if len(sys.argv) > 2 else 'd:/EmpathEase v1/data/bhaav'
    hinglish_csv = sys.argv[3] if len(sys.argv) > 3 else 'd:/EmpathEase v1/data/hinglish_therapy_emotion_dataset_v2.csv'

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            goemotions_dir=goemotions_dir,
            bhaav_dir=bhaav_dir,
            hinglish_csv=hinglish_csv,
            tokenizer_name='google/muril-base-cased',
            batch_size=16
        )

        # Test a batch
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Input IDs: {batch['input_ids'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        print(f"  Label values: {batch['labels'].tolist()}")
        print(f"  Label names: {[EMOTION_LABELS[l] for l in batch['labels'].tolist()]}")

        # Count label distribution in train
        from collections import Counter
        all_labels = []
        for ds in train_loader.dataset.datasets:
            if hasattr(ds, 'labels'):
                all_labels.extend(ds.labels)
        dist = Counter(all_labels)
        print(f"\nTrain label distribution:")
        for idx in range(NUM_LABELS):
            name = EMOTION_LABELS[idx]
            count = dist.get(idx, 0)
            print(f"  {name:>12}: {count:>6}")

    except Exception as e:
        import traceback
        traceback.print_exc()
