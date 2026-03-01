"""
GoEmotions Dataset Loader for Text Emotion Classification.

GoEmotions is a dataset of 58K Reddit comments labeled with 27 emotion categories
plus neutral, for a total of 28 classes. It supports multi-label classification.

Supports loading from:
1. Local TSV files (from Google Research GitHub)
2. HuggingFace datasets (fallback)

Download TSV files from:
https://github.com/google-research/google-research/tree/master/goemotions/data
"""

import os
import csv
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# GoEmotions emotion labels (28 classes)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

# Simplified emotion mapping for EmpathEase (7 core emotions)
SIMPLIFIED_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
SIMPLIFIED_TO_IDX = {label: idx for idx, label in enumerate(SIMPLIFIED_EMOTIONS)}

# Mapping from 28 GoEmotions to 7 simplified
GOEMOTION_TO_SIMPLIFIED = {
    'admiration': 'joy',
    'amusement': 'joy',
    'anger': 'anger',
    'annoyance': 'anger',
    'approval': 'joy',
    'caring': 'joy',
    'confusion': 'neutral',
    'curiosity': 'neutral',
    'desire': 'joy',
    'disappointment': 'sadness',
    'disapproval': 'anger',
    'disgust': 'disgust',
    'embarrassment': 'fear',
    'excitement': 'joy',
    'fear': 'fear',
    'gratitude': 'joy',
    'grief': 'sadness',
    'joy': 'joy',
    'love': 'joy',
    'nervousness': 'fear',
    'optimism': 'joy',
    'pride': 'joy',
    'realization': 'surprise',
    'relief': 'joy',
    'remorse': 'sadness',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'neutral': 'neutral'
}


class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions.
    
    Loads data from local TSV files or HuggingFace.
    Supports both full 28-class and simplified 7-class modes.
    """
    
    def __init__(
        self,
        split: str = 'train',
        tokenizer_name: str = 'distilbert-base-uncased',
        max_length: int = 128,
        simplified: bool = False,
        data_dir: Optional[str] = None,
        use_local: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            split: 'train', 'validation' (or 'dev'), or 'test'
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum token length
            simplified: Use 7-class simplified emotions instead of 28
            data_dir: Directory containing TSV files (for local loading)
            use_local: If True, try to load from local TSV first
        """
        self.split = split
        self.max_length = max_length
        self.simplified = simplified
        self.num_labels = len(SIMPLIFIED_EMOTIONS) if simplified else NUM_LABELS
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Try local first, then HuggingFace
        if use_local and data_dir:
            self._load_from_local(data_dir)
        else:
            self._load_from_huggingface()
    
    def _load_from_local(self, data_dir: str):
        """Load GoEmotions from local TSV files."""
        data_path = Path(data_dir)
        
        # Map split names
        split_map = {
            'train': 'train.tsv',
            'validation': 'dev.tsv',
            'dev': 'dev.tsv',
            'test': 'test.tsv'
        }
        
        filename = split_map.get(self.split, f'{self.split}.tsv')
        filepath = data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"TSV file not found: {filepath}")
        
        print(f"Loading GoEmotions {self.split} from {filepath}...")
        
        self.texts = []
        self.labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    text = row[0]
                    # Labels are comma-separated indices in column 1
                    label_str = row[1] if len(row) > 1 else ''
                    
                    if label_str:
                        label_indices = [int(x) for x in label_str.split(',') if x.strip()]
                    else:
                        label_indices = [27]  # neutral
                    
                    self.texts.append(text)
                    self.labels.append(label_indices)
        
        print(f"  Loaded {len(self.texts)} samples")
    
    def _load_from_huggingface(self):
        """Load GoEmotions dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading GoEmotions {self.split} from HuggingFace...")
        dataset = load_dataset('go_emotions', 'simplified', split=self.split)
        
        self.texts = dataset['text']
        self.labels = dataset['labels']
        
        print(f"  Loaded {len(self.texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label_indices = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create multi-hot label vector
        if self.simplified:
            # Map to simplified emotions
            label_vector = torch.zeros(len(SIMPLIFIED_EMOTIONS))
            for label_idx in label_indices:
                if label_idx < len(EMOTION_LABELS):
                    original_emotion = EMOTION_LABELS[label_idx]
                    simplified_emotion = GOEMOTION_TO_SIMPLIFIED.get(original_emotion, 'neutral')
                    simplified_idx = SIMPLIFIED_TO_IDX[simplified_emotion]
                    label_vector[simplified_idx] = 1.0
        else:
            label_vector = torch.zeros(NUM_LABELS)
            for label_idx in label_indices:
                if label_idx < NUM_LABELS:
                    label_vector[label_idx] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_vector
        }


def get_dataloaders(
    batch_size: int = 32,
    tokenizer_name: str = 'distilbert-base-uncased',
    max_length: int = 128,
    simplified: bool = False,
    num_workers: int = 0,
    data_dir: Optional[str] = None,
    use_local: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders.
    
    Args:
        batch_size: Batch size
        tokenizer_name: HuggingFace tokenizer
        max_length: Maximum token length
        simplified: Use 7-class emotions
        num_workers: Number of data loading workers
        data_dir: Directory containing local TSV files
        use_local: Try local files first
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = GoEmotionsDataset(
        split='train',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        simplified=simplified,
        data_dir=data_dir,
        use_local=use_local
    )
    
    val_dataset = GoEmotionsDataset(
        split='validation',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        simplified=simplified,
        data_dir=data_dir,
        use_local=use_local
    )
    
    test_dataset = GoEmotionsDataset(
        split='test',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        simplified=simplified,
        data_dir=data_dir,
        use_local=use_local
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    
    # Test dataset loading
    print("Testing GoEmotions Dataset...")
    
    # Check if local data dir provided
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if data_dir:
        print(f"\nUsing local data from: {data_dir}")
        train_data = GoEmotionsDataset(
            split='train', 
            simplified=True, 
            data_dir=data_dir,
            use_local=True
        )
    else:
        print("\nUsing HuggingFace (no local data dir provided)")
        train_data = GoEmotionsDataset(split='train', simplified=True, use_local=False)
    
    print(f"\n7-class simplified mode:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Num labels: {train_data.num_labels}")
    
    # Test a sample
    sample = train_data[0]
    print(f"\nSample:")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
