"""
MobileNetV3-based Facial Affect Recognition Model.

This model is designed for both:
- Edge deployment (TensorFlow.js via ONNX)
- Server-side inference (PyTorch/ONNX)

Architecture:
- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Head: Custom classifier for 7 emotions
- Optional: Valence-Arousal dimensional output
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torchvision import models


# Number of emotion classes (unified mapping)
NUM_EMOTIONS = 7
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class FacialAffectModel(nn.Module):
    """
    MobileNetV3-based facial affect recognition model.
    
    Features:
    - Pretrained backbone for transfer learning
    - Dropout for regularization
    - Optional valence-arousal head for dimensional emotions
    """
    
    def __init__(
        self,
        num_classes: int = NUM_EMOTIONS,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_dimensional: bool = False,
        freeze_backbone: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of emotion classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before classifier
            use_dimensional: Add valence-arousal regression head
            freeze_backbone: Freeze backbone weights (for fine-tuning heads only)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_dimensional = use_dimensional
        
        # Load pretrained MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        
        # Extract features (everything except classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Get the number of features from the backbone
        # MobileNetV3-Small has 576 features before the classifier
        num_features = 576
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Custom classifier head for emotions
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, num_classes)
        )
        
        # Optional dimensional emotion head (valence-arousal)
        if use_dimensional:
            self.dimensional_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, 2),  # [valence, arousal]
                nn.Tanh()  # Output in [-1, 1]
            )
    
    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Dictionary with:
            - 'logits': Emotion class logits (B, num_classes)
            - 'probs': Softmax probabilities (B, num_classes)
            - 'dimensional': Valence-arousal (B, 2) if use_dimensional=True
        """
        # Feature extraction
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Emotion classification
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probs': probs
        }
        
        # Optional dimensional output
        if self.use_dimensional:
            dimensional = self.dimensional_head(features)
            output['dimensional'] = dimensional
        
        return output
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted class and confidence.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple of:
            - predicted_classes: (B,) tensor of class indices
            - confidences: (B,) tensor of confidence scores
        """
        with torch.no_grad():
            output = self.forward(x)
            probs = output['probs']
            confidences, predicted = torch.max(probs, dim=1)
            return predicted, confidences
    
    def get_emotion_name(self, class_idx: int) -> str:
        """Get emotion label for class index."""
        return EMOTION_LABELS[class_idx]


class FacialAffectModelLarge(nn.Module):
    """
    EfficientNet-B0 based model for server-side inference.
    
    Larger and more accurate than MobileNetV3, but slower.
    Use this for server-side processing where latency is less critical.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_EMOTIONS,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_dimensional: bool = False,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_dimensional = use_dimensional
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        
        # Extract features
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # EfficientNet-B0 has 1280 features
        num_features = 1280
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Optional dimensional emotion head
        if use_dimensional:
            self.dimensional_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, 2),
                nn.Tanh()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probs': probs
        }
        
        if self.use_dimensional:
            output['dimensional'] = self.dimensional_head(features)
        
        return output
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.forward(x)
            probs = output['probs']
            confidences, predicted = torch.max(probs, dim=1)
            return predicted, confidences


def create_model(
    variant: str = 'mobilenet',
    num_classes: int = NUM_EMOTIONS,
    pretrained: bool = True,
    dropout: float = 0.3,
    use_dimensional: bool = False,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create facial affect model.
    
    Args:
        variant: 'mobilenet' for edge or 'efficientnet' for server
        num_classes: Number of emotion classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        use_dimensional: Include valence-arousal head
        freeze_backbone: Freeze backbone weights
    
    Returns:
        PyTorch model
    """
    if variant == 'mobilenet':
        return FacialAffectModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            use_dimensional=use_dimensional,
            freeze_backbone=freeze_backbone
        )
    elif variant == 'efficientnet':
        return FacialAffectModelLarge(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            use_dimensional=use_dimensional,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'mobilenet' or 'efficientnet'")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # Test model creation
    print("Testing FacialAffectModel (MobileNetV3)...")
    model = create_model('mobilenet', pretrained=True)
    print(f"  Total params: {count_parameters(model, False):,}")
    print(f"  Trainable params: {count_parameters(model, True):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"  Output shapes: logits={output['logits'].shape}, probs={output['probs'].shape}")
    
    # Test prediction
    pred, conf = model.predict(x)
    print(f"  Predictions: {pred}, Confidences: {conf}")
    
    print("\nTesting FacialAffectModelLarge (EfficientNet)...")
    model_large = create_model('efficientnet', pretrained=True)
    print(f"  Total params: {count_parameters(model_large, False):,}")
    print(f"  Trainable params: {count_parameters(model_large, True):,}")
    
    output_large = model_large(x)
    print(f"  Output shapes: logits={output_large['logits'].shape}")
    
    print("\nTesting with dimensional output...")
    model_dim = create_model('mobilenet', use_dimensional=True)
    output_dim = model_dim(x)
    print(f"  Dimensional shape: {output_dim['dimensional'].shape}")
