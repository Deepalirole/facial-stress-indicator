"""
Training script for Facial Stress Indicator.

Usage:
    python -m src.train
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from src.config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE,
    CHECKPOINTS_DIR, LOGS_DIR, BEST_MODEL_PATH, LAST_MODEL_PATH,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    RANDOM_SEED, LOG_INTERVAL, SAVE_CHECKPOINT_INTERVAL
)
from src.dataset import create_dataloaders
from models.mobilenet_stress import get_mobilenet_stress_model
from src.utils import (
    calculate_accuracy, save_checkpoint, save_training_logs
)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        acc = calculate_accuracy(outputs, labels)
        running_acc += acc * batch_size
        total_samples += batch_size
        
        # Update progress bar
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            acc = calculate_accuracy(outputs, labels)
            running_acc += acc * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train Facial Stress Indicator')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("=" * 60)
    print("Facial Stress Indicator - Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
        use_folder_structure=True  # Will auto-detect if folders exist
    )
    
    # Create model
    print("\nCreating model...")
    model, criterion, optimizer, scheduler = get_mobilenet_stress_model(
        learning_rate=args.lr,
        device=device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    
    # Create output directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    training_logs = []
    no_improvement_count = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler if available
        if scheduler:
            scheduler.step()
        
        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'timestamp': datetime.now().isoformat()
        }
        training_logs.append(log_entry)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            no_improvement_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                BEST_MODEL_PATH, is_best=True
            )
        else:
            no_improvement_count += 1
        
        # Save last checkpoint periodically
        if (epoch + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                LAST_MODEL_PATH, is_best=False
            )
        
        # Early stopping
        if no_improvement_count >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break
        
        print("-" * 60)
    
    # Save training logs
    log_file = LOGS_DIR / f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_training_logs(training_logs, log_file, format="json")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Training logs saved to: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()



