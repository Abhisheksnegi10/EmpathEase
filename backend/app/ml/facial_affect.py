"""
Facial Affect Recognition Inference Module.

Server-side inference wrapper for emotion detection from face images.
Uses ONNX runtime for efficient inference.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
from PIL import Image

# Try to import optional dependencies
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Emotion labels (unified mapping)
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Default model paths — LOCKED production model (v1.0.0, 2026-02-20)
BASE_DIR = Path(__file__).parent.parent.parent  # backend/
DEFAULT_ONNX_MODEL = BASE_DIR / 'models' / 'facial_emotion' / 'efficientnet_b0_v1.onnx'  # Future ONNX export
DEFAULT_PYTORCH_MODEL = BASE_DIR / 'models' / 'facial_emotion' / 'efficientnet_b0_v1.pth'

# Preprocessing constants
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FacialAffectInference:
    """
    Facial affect recognition inference engine.
    
    Supports both ONNX and PyTorch models for flexibility.
    ONNX is preferred for production (faster, smaller memory footprint).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_onnx: bool = False,  # Default to PyTorch since we have .pt weights
        device: str = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu',
        lazy_load: bool = True  # Lazy load for faster startup
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file (ONNX or PyTorch)
            use_onnx: Use ONNX runtime (faster)
            device: 'cuda' or 'cpu' (for PyTorch only)
            lazy_load: If True, load model on first prediction
        """
        self.use_onnx = use_onnx and HAS_ONNX
        self.device = device
        self.model = None
        self.session = None
        self._loaded = False
        self.lazy_load = lazy_load
        
        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        elif self.use_onnx:
            self.model_path = DEFAULT_ONNX_MODEL
        else:
            self.model_path = DEFAULT_PYTORCH_MODEL
        
        # Load model (or defer if lazy loading)
        if not self.lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load the inference model."""
        if self._loaded:
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        self._loaded = True
    
    def _load_onnx_model(self):
        """Load ONNX model with ONNX Runtime."""
        print(f"Loading ONNX model: {self.model_path}")
        
        # Configure providers
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device.lower():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        print(f"  Loaded successfully (providers: {self.session.get_providers()})")
    
    def _load_pytorch_model(self):
        """Load PyTorch EfficientNet-B0 model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        import logging
        from torch import nn
        from torchvision import models
        
        logger = logging.getLogger(__name__)
        print(f"Loading PyTorch model: {self.model_path}")
        
        # Define efficientnet architecture matching training script
        # We don't need 'pretrained=True' here since we load custom weights
        self.model = models.efficientnet_b0(weights=None)
        
        # Recreate classifier head
        num_classes = len(EMOTION_LABELS)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
        start_epoch = 0
        best_acc = 0.0
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']  # train_efficientnet.py format
                    if 'epoch' in checkpoint: start_epoch = checkpoint['epoch']
                    if 'best_acc' in checkpoint: best_acc = checkpoint.get('best_acc', 0.0)
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            print(f"  Loaded weights (Epoch {start_epoch}, Acc {best_acc:.2f}%)")
            
        except Exception as e:
            print(f"Error loading state dict: {e}")
            raise e
        
        self.model.to(self.device)
        self.model.eval()
        print(f"  Model ready (device: {self.device})")
    
    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, str]
    ) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
        
        Returns:
            Preprocessed image array of shape (1, 3, 224, 224)
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 4:  # RGBA
                image = Image.fromarray(image).convert('RGB')
            elif len(image.shape) == 2:  # Grayscale
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image)
        
        # Resize
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        
        # Convert to numpy
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - MEAN) / STD
        
        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, 'torch.Tensor'],
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Predict emotions from face image.
        
        Args:
            image: Input face image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with:
            - 'top_emotion': Most likely emotion label
            - 'confidence': Confidence score [0, 1]
            - 'all_probs': Dict of all emotion probabilities
            - 'top_k': List of top-k (emotion, probability) tuples
        """
        # Ensure model is loaded
        self._load_model()
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Inference
        if self.use_onnx:
            probs = self._predict_onnx(input_data)
        else:
            probs = self._predict_pytorch(input_data)
        
        # Process results
        probs = probs[0]  # Remove batch dimension
        
        # Get top emotion
        top_idx = int(np.argmax(probs))
        top_emotion = EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        
        # All probabilities
        all_probs = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        
        # Top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [(EMOTION_LABELS[i], float(probs[i])) for i in top_indices]
        
        return {
            'top_emotion': top_emotion,
            'confidence': confidence,
            'all_probs': all_probs,
            'top_k': top_k_results
        }
    
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[1]  # Return probabilities (second output)
    
    def _predict_pytorch(self, input_data: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        import torch.nn.functional as F
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).to(self.device)
            output = self.model(input_tensor)
            # Model returns raw logits tensor (not a dict)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('probs', list(output.values())[0]))
            else:
                logits = output
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        batch_size: int = 16
    ) -> List[Dict[str, any]]:
        """
        Batch prediction for multiple images.
        
        Args:
            images: List of input images
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_data = np.concatenate([self.preprocess(img) for img in batch], axis=0)
            
            # Inference
            if self.use_onnx:
                probs = self.session.run(None, {self.input_name: batch_data})[1]
            else:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(batch_data).to(self.device)
                    probs = self.model(input_tensor)['probs'].cpu().numpy()
            
            # Process each result
            for j in range(len(batch)):
                p = probs[j]
                top_idx = int(np.argmax(p))
                results.append({
                    'top_emotion': EMOTION_LABELS[top_idx],
                    'confidence': float(p[top_idx]),
                    'all_probs': {label: float(p[k]) for k, label in enumerate(EMOTION_LABELS)}
                })
        
        return results


# Singleton instance for easy import
_inference_engine: Optional[FacialAffectInference] = None


def get_inference_engine(
    model_path: Optional[str] = None,
    use_onnx: bool = False  # Default to PyTorch since we only have .pt weights
) -> FacialAffectInference:
    """
    Get or create the facial affect inference engine.
    
    Args:
        model_path: Optional path to model file
        use_onnx: Use ONNX runtime
    
    Returns:
        FacialAffectInference instance
    """
    global _inference_engine
    
    if _inference_engine is None:
        _inference_engine = FacialAffectInference(
            model_path=model_path,
            use_onnx=use_onnx
        )
    
    return _inference_engine


def predict_emotion(
    image: Union[np.ndarray, Image.Image, str]
) -> Dict[str, any]:
    """
    Convenience function to predict emotion from image.
    
    Args:
        image: Input face image
    
    Returns:
        Prediction dictionary
    """
    engine = get_inference_engine()
    return engine.predict(image)


if __name__ == '__main__':
    # Test inference
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python facial_affect.py <image_path>")
        print("\nThis will test the facial affect inference on the provided image.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Testing facial affect inference on: {image_path}")
    
    try:
        engine = FacialAffectInference(use_onnx=True)
        result = engine.predict(image_path)
        
        print(f"\nPrediction:")
        print(f"  Top emotion: {result['top_emotion']} ({result['confidence']:.1%})")
        print(f"\n  All probabilities:")
        for emotion, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"    {emotion:12s} {prob:5.1%} {bar}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to train and export the model first:")
        print("  1. python -m training.facial_affect.train")
        print("  2. python -m training.facial_affect.export --checkpoint outputs/facial_affect/model_best.pt")
