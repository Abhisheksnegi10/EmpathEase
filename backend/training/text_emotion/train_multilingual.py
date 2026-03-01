"""
Train Text Emotion v2.1 — 8-class MuRIL fine-tune.

Key changes from v1:
  - CrossEntropyLoss (not BCE — single-label multi-class)
  - Batch 16 + gradient accumulation 2 (GTX 1650 4GB VRAM)
  - Freeze bottom 6/12 MuRIL encoder layers
  - Linear warmup + cosine decay LR schedule
  - Checkpoint best macro F1 per epoch
  - 8 classes: anger, disgust, fear, joy, sadness, surprise, neutral, suppressed
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
import numpy as np

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.text_emotion.dataset_multilingual import (
    get_dataloaders,
    EMOTION_LABELS,
    NUM_LABELS,
)


# ============================================================================
# Args
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Train Text Emotion v2.1")
    # Data
    parser.add_argument('--goemotions_dir', type=str,
                        default='d:/EmpathEase v1/data/goemotions')
    parser.add_argument('--bhaav_dir', type=str,
                        default='d:/EmpathEase v1/data/bhaav')
    parser.add_argument('--hinglish_csv', type=str,
                        default='d:/EmpathEase v1/data/hinglish_therapy_emotion_dataset_v2.csv')
    # Model
    parser.add_argument('--model_name', type=str,
                        default='google/muril-base-cased')
    parser.add_argument('--max_length', type=int, default=128)
    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--freeze_layers', type=int, default=6,
                        help='Freeze bottom N of 12 MuRIL encoder layers')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mixed precision (may cause CUBLAS errors on GTX 1650)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='d:/EmpathEase v1/backend/outputs/text_emotion_v2/best')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='d:/EmpathEase v1/backend/outputs/text_emotion_v2/checkpoints')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume from this epoch (0 = start fresh)')
    return parser.parse_args()


# ============================================================================
# Class weights (from v2.1 plan)
# ============================================================================

CLASS_WEIGHTS = torch.tensor([
    1.0,   # anger     — well-represented
    2.5,   # disgust   — weak class
    2.5,   # fear      — weak class
    1.0,   # joy       — well-represented
    1.2,   # sadness   — slight underperformance
    2.5,   # surprise  — weak class
    1.0,   # neutral   — well-represented
    3.0,   # suppressed — new class, smallest signal
])


# ============================================================================
# Helpers
# ============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def freeze_layers(model, num_freeze=6):
    """Freeze bottom N encoder layers of MuRIL (BERT)."""
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze bottom N encoder layers
    for i in range(num_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen {num_freeze}/12 layers. Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


# ============================================================================
# Train one epoch
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum=2):
    model.train()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum  # Scale for accumulation

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum, input_ids.size(0))

        # Collect predictions for F1
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss_meter.avg:.4f} | LR: {lr:.2e}")

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return loss_meter.avg, macro_f1


# ============================================================================
# Validate
# ============================================================================

def validate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_meter.update(outputs.loss.item(), input_ids.size(0))

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0,
                         labels=list(range(NUM_LABELS)))

    return loss_meter.avg, macro_f1, per_class, all_preds, all_labels


# ============================================================================
# Main
# ============================================================================

def main():
    args = get_args()

    print("=" * 70)
    print("  Text Emotion v2.1 Training")
    print(f"  Model: {args.model_name}")
    print(f"  Labels: {NUM_LABELS} classes: {EMOTION_LABELS}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr} (warmup {args.warmup_ratio*100:.0f}%)")
    print(f"  Freeze: bottom {args.freeze_layers}/12 layers")
    print(f"  FP16: {args.fp16}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data ---
    print("\n--- Loading Datasets ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        goemotions_dir=args.goemotions_dir,
        bhaav_dir=args.bhaav_dir,
        hinglish_csv=args.hinglish_csv,
        batch_size=args.batch_size,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # --- Model ---
    print("\n--- Loading Model ---")
    config = AutoConfig.from_pretrained(args.model_name, num_labels=NUM_LABELS)
    config.id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
    config.label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    config.problem_type = "single_label_classification"  # CrossEntropy

    # --- Resume or fresh load ---
    start_epoch = args.resume_epoch
    if start_epoch > 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{start_epoch}")
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}...")
            model = BertForSequenceClassification.from_pretrained(ckpt_path).to(device)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {ckpt_path}, starting fresh")
            start_epoch = 0
            model = BertForSequenceClassification.from_pretrained(
                args.model_name, config=config
            ).to(device)
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, config=config
        ).to(device)

    # --- Freeze layers ---
    freeze_layers(model, args.freeze_layers)

    # --- Loss (CrossEntropyLoss with class weights) ---
    weights = CLASS_WEIGHTS.to(device)

    # --- Pre-load tokenizer (once, to avoid re-downloading during saves) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- Optimizer (MUST be after resume so it references correct params) ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    # --- Scheduler (warmup + cosine) ---
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # --- Scaler (FP16) ---
    scaler = GradScaler() if args.fp16 and device.type == 'cuda' else None

    # --- Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Custom loss with class weights ---
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --- Training loop ---
    best_macro_f1 = 0.0
    training_log = []

    print("\n--- Training ---")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_loss, train_f1 = train_epoch_weighted(
            model, train_loader, optimizer, scheduler, scaler,
            device, loss_fn, args.grad_accum
        )
        epoch_time = time.time() - epoch_start

        # Validate
        val_loss, val_f1, per_class_f1, val_preds, val_labels = validate_weighted(
            model, val_loader, device, loss_fn
        )

        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'per_class_f1': {EMOTION_LABELS[i]: float(per_class_f1[i]) for i in range(NUM_LABELS)},
            'time_sec': epoch_time,
        }
        training_log.append(log_entry)

        print(f"\n  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")
        print(f"  Time: {epoch_time:.0f}s")
        print(f"\n  Per-class F1:")
        for i, label in enumerate(EMOTION_LABELS):
            status = "[OK]" if per_class_f1[i] >= 0.60 else "[WEAK]" if per_class_f1[i] >= 0.40 else "[BAD]"
            bar = "#" * int(per_class_f1[i] * 20)
            print(f"    {label:>12}: {per_class_f1[i]:.3f} {status:>6}  {bar}")

        # Clear CUDA cache before saving (prevents OOM during save)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Best model — copy from checkpoint instead of re-serializing (saves ~950MB RAM)
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            import shutil
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            shutil.copytree(ckpt_path, args.output_dir)
            print(f"  >>> NEW BEST: macro F1 = {val_f1:.4f} copied to {args.output_dir}")

    # --- Final report ---
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Best macro F1: {best_macro_f1:.4f}")
    print(f"  Saved to: {args.output_dir}")
    print("=" * 70)

    # Save training log
    log_path = os.path.join(args.output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Final classification report on test set
    if test_loader:
        print("\n--- Test Set Evaluation ---")
        test_loss, test_f1, test_per_class, test_preds, test_labels = validate_weighted(
            model, test_loader, device, loss_fn
        )
        print(f"Test macro F1: {test_f1:.4f}")
        print(classification_report(
            test_labels, test_preds,
            target_names=EMOTION_LABELS,
            zero_division=0
        ))


# ============================================================================
# Weighted loss versions of train/validate
# ============================================================================

def train_epoch_weighted(model, loader, optimizer, scheduler, scaler, device, loss_fn, grad_accum=2):
    model.train()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels) / grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum, input_ids.size(0))

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 200 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss_meter.avg:.4f} | LR: {lr:.2e}")

        # Periodic GPU memory cleanup to prevent fragmentation
        if (step + 1) % 500 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return loss_meter.avg, macro_f1


def validate_weighted(model, loader, device, loss_fn):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss_meter.update(loss.item(), input_ids.size(0))

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0,
                         labels=list(range(NUM_LABELS)))

    return loss_meter.avg, macro_f1, per_class, all_preds, all_labels


if __name__ == '__main__':
    main()
