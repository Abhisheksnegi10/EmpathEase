"""
Facial Affect Recognition Training Package.

Trains MobileNetV3 for edge deployment (TensorFlow.js)
and EfficientNet for server-side inference.

Components:
- dataset: PyTorch Dataset classes for FER-2013, AffectNet, CK+
- model: MobileNetV3/EfficientNet architectures
- train: Training script with AMP, scheduling, early stopping
- export: ONNX and TF.js export utilities
"""

from .dataset import (
    FER2013Dataset,
    AffectNetDataset,
    CKPlusDataset,
    get_combined_dataset,
    get_train_transforms,
    get_val_transforms,
    UNIFIED_EMOTIONS
)

from .model import (
    FacialAffectModel,
    FacialAffectModelLarge,
    create_model,
    EMOTION_LABELS
)

__all__ = [
    'FER2013Dataset',
    'AffectNetDataset',
    'CKPlusDataset',
    'get_combined_dataset',
    'get_train_transforms',
    'get_val_transforms',
    'UNIFIED_EMOTIONS',
    'FacialAffectModel',
    'FacialAffectModelLarge',
    'create_model',
    'EMOTION_LABELS'
]
