"""
PyTorch Dataset classes for Facial Affect Recognition.

Supports:
- FER-2013: Folder-based structure (7 emotions)
- AffectNet: YOLO format (8 emotions)
- CK+: Folder-based structure (7 emotions)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T


# Emotion label mappings
FER2013_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
AFFECTNET_EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CK_EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Unified emotion mapping (7 classes - contempt is rare, merge with neutral)
UNIFIED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
UNIFIED_EMOTION_TO_IDX = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}


def map_fer2013_to_unified(label: str) -> int:
    """Map FER-2013 label to unified index."""
    return UNIFIED_EMOTION_TO_IDX[label.lower()]


def map_affectnet_to_unified(label_idx: int) -> int:
    """Map AffectNet label index to unified index."""
    affectnet_to_unified = {
        0: UNIFIED_EMOTION_TO_IDX['angry'],     # Anger
        1: UNIFIED_EMOTION_TO_IDX['neutral'],   # Contempt -> Neutral (rare class)
        2: UNIFIED_EMOTION_TO_IDX['disgust'],   # Disgust
        3: UNIFIED_EMOTION_TO_IDX['fear'],      # Fear
        4: UNIFIED_EMOTION_TO_IDX['happy'],     # Happy
        5: UNIFIED_EMOTION_TO_IDX['neutral'],   # Neutral
        6: UNIFIED_EMOTION_TO_IDX['sad'],       # Sad
        7: UNIFIED_EMOTION_TO_IDX['surprise'],  # Surprise
    }
    return affectnet_to_unified[label_idx]


def map_ck_to_unified(label: str) -> int:
    """Map CK+ label to unified index."""
    ck_to_unified = {
        'anger': UNIFIED_EMOTION_TO_IDX['angry'],
        'contempt': UNIFIED_EMOTION_TO_IDX['neutral'],  # Contempt -> Neutral
        'disgust': UNIFIED_EMOTION_TO_IDX['disgust'],
        'fear': UNIFIED_EMOTION_TO_IDX['fear'],
        'happy': UNIFIED_EMOTION_TO_IDX['happy'],
        'sadness': UNIFIED_EMOTION_TO_IDX['sad'],
        'surprise': UNIFIED_EMOTION_TO_IDX['surprise'],
    }
    return ck_to_unified[label.lower()]


class FER2013Dataset(Dataset):
    """
    FER-2013 Dataset with folder-based structure.
    
    Structure:
        fer 2013/
            train/
                angry/
                disgust/
                ...
            test/
                angry/
                ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.image_size = image_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load all samples
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels."""
        for emotion_folder in FER2013_EMOTIONS:
            emotion_dir = self.split_dir / emotion_folder
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            label = map_fer2013_to_unified(emotion_folder)
            
            for img_path in emotion_dir.glob('*.jpg'):
                self.samples.append((img_path, label))
            for img_path in emotion_dir.glob('*.png'):
                self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from FER-2013 {self.split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AffectNetDataset(Dataset):
    """
    AffectNet Dataset in YOLO format.
    
    Structure:
        Affectnet/
            train/
                images/
                    *.jpg
                labels/
                    *.txt (class_idx center_x center_y width height)
            valid/
            test/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.images_dir = self.root_dir / split / 'images'
        self.labels_dir = self.root_dir / split / 'labels'
        self.transform = transform
        self.image_size = image_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load all samples
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels from YOLO format."""
        if not self.images_dir.exists():
            print(f"Warning: {self.images_dir} does not exist")
            return
        
        for img_path in self.images_dir.glob('*.jpg'):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Parse YOLO label file
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            affectnet_label = int(parts[0])
                            unified_label = map_affectnet_to_unified(affectnet_label)
                            self.samples.append((img_path, unified_label))
            except Exception as e:
                print(f"Error parsing {label_path}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from AffectNet {self.split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CKPlusDataset(Dataset):
    """
    CK+ Dataset with folder-based structure.
    
    Structure:
        CK+/
            anger/
            contempt/
            disgust/
            fear/
            happy/
            sadness/
            surprise/
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load all samples
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels."""
        for emotion_folder in CK_EMOTIONS:
            emotion_dir = self.root_dir / emotion_folder
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            label = map_ck_to_unified(emotion_folder)
            
            for img_path in emotion_dir.glob('*.png'):
                self.samples.append((img_path, label))
            for img_path in emotion_dir.glob('*.jpg'):
                self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from CK+")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms(image_size: int = 224) -> Callable:
    """Get training augmentation transforms."""
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = 224) -> Callable:
    """Get validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_combined_dataset(
    data_root: str,
    split: str = 'train',
    image_size: int = 224,
    include_fer2013: bool = True,
    include_affectnet: bool = True,
    include_ckplus: bool = True,
) -> Dataset:
    """
    Create a combined dataset from multiple sources.
    
    Args:
        data_root: Root directory containing all datasets
        split: 'train', 'valid', or 'test'
        image_size: Target image size
        include_fer2013: Include FER-2013 dataset
        include_affectnet: Include AffectNet dataset
        include_ckplus: Include CK+ dataset (only for training, small dataset)
    
    Returns:
        Combined PyTorch Dataset
    """
    datasets = []
    transform = get_train_transforms(image_size) if split == 'train' else get_val_transforms(image_size)
    
    if include_fer2013:
        fer_split = 'train' if split == 'train' else 'test'
        fer_path = Path(data_root) / 'fer 2013'
        if fer_path.exists():
            datasets.append(FER2013Dataset(str(fer_path), fer_split, transform, image_size))
    
    if include_affectnet:
        affectnet_split_map = {'train': 'train', 'valid': 'valid', 'test': 'test'}
        affectnet_split = affectnet_split_map.get(split, 'train')
        affectnet_path = Path(data_root) / 'Affectnet'
        if affectnet_path.exists():
            datasets.append(AffectNetDataset(str(affectnet_path), affectnet_split, transform, image_size))
    
    if include_ckplus and split == 'train':
        # CK+ is small, only use for training augmentation
        ck_path = Path(data_root) / 'CK+'
        if ck_path.exists():
            datasets.append(CKPlusDataset(str(ck_path), transform, image_size))
    
    if not datasets:
        raise ValueError(f"No datasets found in {data_root}")
    
    return ConcatDataset(datasets)


if __name__ == '__main__':
    # Test dataset loading
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'd:/EmpathEase v1/data'
    
    print("Testing FER-2013 Dataset...")
    fer_train = FER2013Dataset(f'{data_root}/fer 2013', 'train')
    print(f"  Train samples: {len(fer_train)}")
    
    print("\nTesting AffectNet Dataset...")
    affectnet_train = AffectNetDataset(f'{data_root}/Affectnet', 'train')
    print(f"  Train samples: {len(affectnet_train)}")
    
    print("\nTesting CK+ Dataset...")
    ck_dataset = CKPlusDataset(f'{data_root}/CK+')
    print(f"  Total samples: {len(ck_dataset)}")
    
    print("\nTesting Combined Dataset...")
    combined = get_combined_dataset(data_root, 'train')
    print(f"  Combined train samples: {len(combined)}")
    
    # Test loading a sample
    if len(combined) > 0:
        sample, label = combined[0]
        print(f"\nSample shape: {sample.shape}, Label: {UNIFIED_EMOTIONS[label]}")
