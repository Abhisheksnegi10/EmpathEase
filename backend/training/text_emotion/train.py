"""
Training Script for Text Emotion Classification.

Fine-tunes DistilBERT on GoEmotions dataset for multi-label emotion classification.

Features:
- Multi-label classification (BCEWithLogitsLoss)
- Mixed precision training
- Early stopping
- Model checkpointing
- Per-class F1 score evaluation
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.text_emotion.dataset import (
    GoEmotionsDataset,
    get_dataloaders,
    EMOTION_LABELS,
    SIMPLIFIED_EMOTIONS,
    NUM_LABELS
)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Text Emotion Classifier')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                        help='HuggingFace model name')
    parser.add_argument('--simplified', action='store_true',
                        help='Use 7-class simplified emotions instead of 28')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum token length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Training control
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--exp-name', type=str, default='text_emotion',
                        help='Experiment name')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing local TSV files (train.tsv, dev.tsv, test.tsv)')
    parser.add_argument('--use-local', action='store_true',
                        help='Use local TSV files instead of HuggingFace')
    
    # Debug arguments
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick test run')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops training if validation F1 doesn't improve."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_f1: float) -> bool:
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0
        
        return self.early_stop


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    emotion_labels: list = None
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.
    
    Args:
        predictions: Sigmoid probabilities (N, num_classes)
        labels: Ground truth labels (N, num_classes)
        threshold: Classification threshold
        emotion_labels: List of emotion names
    
    Returns:
        Dictionary of metrics
    """
    # Binarize predictions
    preds_binary = (predictions >= threshold).astype(int)
    labels_binary = labels.astype(int)
    
    # Overall metrics
    tp = np.sum((preds_binary == 1) & (labels_binary == 1))
    fp = np.sum((preds_binary == 1) & (labels_binary == 0))
    fn = np.sum((preds_binary == 0) & (labels_binary == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Per-class F1 (optional)
    if emotion_labels:
        per_class_f1 = {}
        for i, label in enumerate(emotion_labels):
            class_tp = np.sum((preds_binary[:, i] == 1) & (labels_binary[:, i] == 1))
            class_fp = np.sum((preds_binary[:, i] == 1) & (labels_binary[:, i] == 0))
            class_fn = np.sum((preds_binary[:, i] == 0) & (labels_binary[:, i] == 1))
            
            class_prec = class_tp / (class_tp + class_fp + 1e-8)
            class_rec = class_tp / (class_tp + class_fn + 1e-8)
            class_f1 = 2 * class_prec * class_rec / (class_prec + class_rec + 1e-8)
            per_class_f1[label] = class_f1
        
        metrics['per_class_f1'] = per_class_f1
    
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_meter.update(loss.item(), input_ids.size(0))
    
    return loss_meter.avg


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    emotion_labels: list = None
) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = nn.BCEWithLogitsLoss()(logits, labels)
            
            loss_meter.update(loss.item(), input_ids.size(0))
            
            # Collect predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = compute_metrics(all_preds, all_labels, threshold, emotion_labels)
    
    return loss_meter.avg, metrics


def save_model(
    model: nn.Module,
    tokenizer,
    output_dir: Path,
    is_best: bool = False
):
    """Save model and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latest
    model.save_pretrained(output_dir / 'latest')
    tokenizer.save_pretrained(output_dir / 'latest')
    
    # Save best
    if is_best:
        model.save_pretrained(output_dir / 'best')
        tokenizer.save_pretrained(output_dir / 'best')


def main():
    """Main training function."""
    args = get_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU for training")
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Determine number of labels
    num_labels = len(SIMPLIFIED_EMOTIONS) if args.simplified else NUM_LABELS
    emotion_labels = SIMPLIFIED_EMOTIONS if args.simplified else EMOTION_LABELS
    
    print(f"\nUsing {'simplified (7-class)' if args.simplified else 'full (28-class)'} mode")
    print(f"Num labels: {num_labels}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        simplified=args.simplified,
        num_workers=args.workers,
        data_dir=args.data_dir,
        use_local=args.use_local
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print()
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, device, args.threshold, emotion_labels
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            print(f"  *** New best F1: {best_f1:.4f} ***")
        
        save_model(model, tokenizer, output_dir, is_best)
        
        # Early stopping check
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = validate(
        model, test_loader, device, args.threshold, emotion_labels
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    if 'per_class_f1' in test_metrics:
        print(f"\nPer-class F1:")
        for label, f1 in sorted(test_metrics['per_class_f1'].items(), key=lambda x: -x[1]):
            print(f"  {label}: {f1:.4f}")
    
    print(f"\nTraining complete!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
