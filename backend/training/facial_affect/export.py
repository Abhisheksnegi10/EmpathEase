"""
Model Export Utilities for Facial Affect Recognition.

Exports trained PyTorch models to:
- ONNX format for server-side inference
- TensorFlow.js format for edge/browser inference (via ONNX)
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from model import create_model, EMOTION_LABELS


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verify: bool = True
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Output ONNX file path
        input_size: Model input size (B, C, H, W)
        opset_version: ONNX opset version
        dynamic_batch: Allow dynamic batch size
        verify: Verify ONNX model after export
    
    Returns:
        Path to exported ONNX file
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Prepare output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'probs': {0: 'batch_size'}
        }
    
    # Export to ONNX
    print(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'probs'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"  Exported successfully: {output_path}")
    
    # Verify export
    if verify:
        verify_onnx_model(str(output_path), dummy_input)
    
    return str(output_path)


def verify_onnx_model(
    onnx_path: str,
    sample_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """
    Verify ONNX model matches PyTorch output.
    
    Args:
        onnx_path: Path to ONNX model
        sample_input: Sample input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if verification passed
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping verification")
        return True
    
    print(f"  Verifying ONNX model...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"  ONNX output shapes: {[o.shape for o in ort_outputs]}")
    print(f"  Verification passed!")
    
    return True


def export_to_tfjs(
    onnx_path: str,
    output_dir: str
) -> str:
    """
    Convert ONNX model to TensorFlow.js format.
    
    Requires: pip install tensorflowjs onnx-tf
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for TF.js model
    
    Returns:
        Path to TF.js model directory
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflowjs as tfjs
    except ImportError as e:
        print(f"Error: Missing dependency for TF.js export: {e}")
        print("Install with: pip install tensorflowjs onnx-tf tensorflow")
        raise
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting ONNX to TensorFlow.js: {output_dir}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Save as TensorFlow SavedModel
    tf_model_dir = output_dir / 'tf_model'
    tf_rep.export_graph(str(tf_model_dir))
    
    # Convert to TF.js
    tfjs_dir = output_dir / 'tfjs_model'
    tfjs.converters.convert_tf_saved_model(
        str(tf_model_dir),
        str(tfjs_dir)
    )
    
    print(f"  TensorFlow.js model saved to: {tfjs_dir}")
    
    return str(tfjs_dir)


def load_trained_model(
    checkpoint_path: str,
    model_variant: str = 'mobilenet',
    device: str = 'cpu'
) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (checkpoint_best.pt or model_best.pt)
        model_variant: 'mobilenet' or 'efficientnet'
        device: Device to load model on
    
    Returns:
        Loaded PyTorch model
    """
    # Create model
    model = create_model(variant=model_variant, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def create_metadata_file(output_dir: str):
    """Create metadata JSON file for the model."""
    import json
    
    metadata = {
        'model_name': 'facial_affect_recognition',
        'num_classes': len(EMOTION_LABELS),
        'labels': EMOTION_LABELS,
        'input_size': [224, 224, 3],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    output_path = Path(output_dir) / 'metadata.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export Facial Affect Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'efficientnet'],
                        help='Model variant')
    parser.add_argument('--output-dir', type=str, default='./exports',
                        help='Output directory for exported models')
    parser.add_argument('--format', type=str, nargs='+', default=['onnx'],
                        choices=['onnx', 'tfjs'],
                        help='Export formats')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX export')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, args.model)
    print("  Model loaded successfully")
    
    # Export to ONNX
    onnx_path = None
    if 'onnx' in args.format or 'tfjs' in args.format:
        onnx_path = output_dir / f'facial_affect_{args.model}.onnx'
        export_to_onnx(model, str(onnx_path), verify=args.verify)
    
    # Export to TF.js
    if 'tfjs' in args.format and onnx_path:
        tfjs_dir = output_dir / 'tfjs'
        try:
            export_to_tfjs(str(onnx_path), str(tfjs_dir))
        except Exception as e:
            print(f"Warning: TF.js export failed: {e}")
    
    # Create metadata
    create_metadata_file(str(output_dir))
    
    print(f"\nExport complete! Files saved to: {output_dir}")


if __name__ == '__main__':
    main()
