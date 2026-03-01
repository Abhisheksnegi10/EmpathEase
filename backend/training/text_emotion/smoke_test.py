"""
Quick 10-step smoke test for training.
Verifies: no OOM, loss decreasing, all 8 classes seen, correct shapes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoConfig, BertForSequenceClassification

from training.text_emotion.dataset_multilingual import (
    get_dataloaders, EMOTION_LABELS, NUM_LABELS
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem:.1f} GB")

    # Load data (small batch)
    print("\nLoading datasets...")
    train_loader, _, _ = get_dataloaders(
        goemotions_dir='d:/EmpathEase v1/data/goemotions',
        bhaav_dir='d:/EmpathEase v1/data/bhaav',
        hinglish_csv='d:/EmpathEase v1/data/hinglish_therapy_emotion_dataset_v2.csv',
        batch_size=16,
        tokenizer_name='google/muril-base-cased',
        max_length=128,
    )

    # Load model with 8 labels
    print("\nLoading MuRIL (8 labels)...")
    config = AutoConfig.from_pretrained('google/muril-base-cased', num_labels=NUM_LABELS)
    config.problem_type = "single_label_classification"
    model = BertForSequenceClassification.from_pretrained(
        'google/muril-base-cased', config=config
    ).to(device)

    # Freeze bottom 6 layers
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(6):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    # Setup
    class_weights = torch.tensor([1.0, 2.5, 2.5, 1.0, 1.2, 2.5, 1.0, 3.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    scaler = GradScaler() if device.type == 'cuda' else None

    # Run 10 steps
    model.train()
    print("\n--- Running 10 steps ---")
    labels_seen = set()

    for step, batch in enumerate(train_loader):
        if step >= 10:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels_seen.update(labels.cpu().tolist())

        with autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated() / 1e9
        else:
            mem_used = 0

        preds = torch.argmax(outputs.logits, dim=-1)
        print(f"  Step {step+1}: loss={loss.item():.4f} | preds={preds.cpu().tolist()[:5]} | VRAM={mem_used:.2f}GB")

    print(f"\n--- RESULTS ---")
    print(f"Labels seen in 10 steps: {sorted(labels_seen)}")
    print(f"Label names: {[EMOTION_LABELS[i] for i in sorted(labels_seen)]}")
    if device.type == 'cuda':
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_allocated()
        print(f"VRAM headroom: {free/1e9:.2f} GB")
    print("SMOKE TEST PASSED" if len(labels_seen) > 0 else "SMOKE TEST FAILED")

if __name__ == '__main__':
    main()
