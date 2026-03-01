"""
Training Script for Facial Affect Recognition.

Features:
- Mixed precision training (AMP)
- Learning rate scheduling (OneCycleLR)
- Early stopping
- Model checkpointing
- TensorBoard logging
- Gradient accumulation for large effective batch sizes
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.facial_affect.dataset import (
    get_combined_dataset,
    get_train_transforms,
    get_val_transforms,
    UNIFIED_EMOTIONS
)
from training.facial_affect.model import create_model, count_parameters


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Facial Affect Recognition Model')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='d:/EmpathEase v1/data',
                        help='Root directory containing datasets')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--use-fer2013', action='store_true', default=True,
                        help='Use FER-2013 dataset')
    parser.add_argument('--use-affectnet', action='store_true', default=True,
                        help='Use AffectNet dataset')
    parser.add_argument('--use-ckplus', action='store_true', default=True,
                        help='Use CK+ dataset')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'efficientnet'],
                        help='Model variant')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone for initial epochs')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training control
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--exp-name', type=str, default='facial_affect',
                        help='Experiment name')
    
    # Debug arguments
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick test run with minimal data')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only N samples for testing')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
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
    """Early stops training if validation accuracy doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc: float) -> bool:
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def compute_class_weights(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    class_counts = np.zeros(len(UNIFIED_EMOTIONS))
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_accum: int = 1
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs['logits'], labels)
            loss = loss / grad_accum  # Scale for gradient accumulation
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Compute accuracy
        _, predicted = torch.max(outputs['logits'], 1)
        correct = (predicted == labels).sum().item()
        
        loss_meter.update(loss.item() * grad_accum, images.size(0))
        acc_meter.update(correct / images.size(0), images.size(0))
    
    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Per-class accuracy
    class_correct = np.zeros(len(UNIFIED_EMOTIONS))
    class_total = np.zeros(len(UNIFIED_EMOTIONS))
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs['logits'], labels)
            
            _, predicted = torch.max(outputs['logits'], 1)
            correct = (predicted == labels).sum().item()
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(correct / images.size(0), images.size(0))
            
            # Per-class stats
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Compute per-class accuracy
    per_class_acc = {}
    for i, emotion in enumerate(UNIFIED_EMOTIONS):
        if class_total[i] > 0:
            per_class_acc[emotion] = class_correct[i] / class_total[i]
        else:
            per_class_acc[emotion] = 0.0
    
    return loss_meter.avg, acc_meter.avg, per_class_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_acc: float,
    output_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latest
    torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
    
    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
        # Also save model-only for inference
        torch.save(model.state_dict(), output_dir / 'model_best.pt')
    
    # Save periodic
    if epoch % 5 == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


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
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = get_combined_dataset(
        args.data_root,
        split='train',
        image_size=args.image_size,
        include_fer2013=args.use_fer2013,
        include_affectnet=args.use_affectnet,
        include_ckplus=args.use_ckplus
    )
    
    val_dataset = get_combined_dataset(
        args.data_root,
        split='valid',
        image_size=args.image_size,
        include_fer2013=args.use_fer2013,
        include_affectnet=args.use_affectnet,
        include_ckplus=False  # CK+ is too small for validation
    )
    
    # Subset for testing
    if args.subset:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(args.subset, len(train_dataset)))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(args.subset // 4, len(val_dataset)))
        )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(
        variant=args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    print(f"  Total params: {count_parameters(model, False):,}")
    print(f"  Trainable params: {count_parameters(model, True):,}")
    
    # Compute class weights for imbalanced data
    print("\nComputing class weights...")
    # Note: This can be slow for large datasets, skip if needed
    # class_weights = compute_class_weights(train_dataset).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_accum
        )
        
        # Step scheduler
        scheduler.step()
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        for emotion, acc in per_class_acc.items():
            writer.add_scalar(f'PerClass/{emotion}', acc, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Per-class: {' | '.join([f'{e[:3]}:{a:.2f}' for e, a in per_class_acc.items()])}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  *** New best validation accuracy: {best_val_acc:.4f} ***")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, output_dir, is_best
        )
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Final save
    print(f"\nTraining complete!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    
    writer.close()


if __name__ == '__main__':
    main()
