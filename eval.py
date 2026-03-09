"""
Evaluation script for Facial Stress Indicator.

Evaluates model on test set and generates metrics, confusion matrix, etc.

Usage:
    python -m src.eval --checkpoint outputs/checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from tqdm import tqdm
import json

from src.config import (
    DEVICE, BEST_MODEL_PATH, METRICS_DIR, CONFUSION_MATRICES_DIR,
    CLASS_NAMES, NUM_CLASSES
)
from src.dataset import create_dataloaders
from models.mobilenet_stress import load_model
from src.utils import calculate_accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: list
) -> dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            acc = calculate_accuracy(outputs, labels)
            running_acc += acc * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    test_loss = running_loss / total_samples
    test_acc = running_acc / total_samples
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'overall_accuracy': float(overall_acc),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, cm, all_preds, all_labels


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_metrics(metrics: dict, save_path: Path):
    """Save metrics to JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path}")


def print_metrics(metrics: dict, class_names: list):
    """Print metrics to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} "
              f"{metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} "
              f"{metrics['per_class_f1'][i]:<12.4f} "
              f"{int(metrics['per_class_support'][i]):<10}")
    
    print("\nClassification Report:")
    print("-" * 60)
    from sklearn.metrics import classification_report
    # Reconstruct labels and preds for report (we'll use the stored metrics)
    print(f"\nMacro Avg: Precision={np.mean(metrics['per_class_precision']):.4f}, "
          f"Recall={np.mean(metrics['per_class_recall']):.4f}, "
          f"F1={np.mean(metrics['per_class_f1']):.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Facial Stress Indicator')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(BEST_MODEL_PATH),
        help='Path to model checkpoint'
    )
    parser.add_argument('--device', type=str, default=DEVICE, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("=" * 60)
    print("Facial Stress Indicator - Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        use_folder_structure=True
    )
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model(str(checkpoint_path), device=device)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics, cm, all_preds, all_labels = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = METRICS_DIR / f"metrics_{checkpoint_path.stem}.json"
    save_metrics(metrics, metrics_file)
    
    # Save confusion matrix
    CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    cm_file = CONFUSION_MATRICES_DIR / f"confusion_matrix_{checkpoint_path.stem}.png"
    plot_confusion_matrix(cm, class_names, cm_file)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Confusion matrix saved to: {cm_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()



