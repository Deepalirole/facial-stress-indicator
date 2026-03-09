"""
Utility functions for training, evaluation, and inference.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import csv

from src.config import CLASS_NAMES, REVERSE_CLASS_MAP


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy given predictions and targets.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
    
    Returns:
        Accuracy as float
    """
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == targets).item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: Path,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'is_best': is_best
    }
    torch.save(checkpoint, filepath)
    if is_best:
        print(f"Saved best model checkpoint to {filepath}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cuda"
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load on
    
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_training_logs(
    logs: List[Dict],
    filepath: Path,
    format: str = "csv"
):
    """
    Save training logs to file.
    
    Args:
        logs: List of log dictionaries
        filepath: Path to save logs
        format: Format to save in ('csv' or 'json')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        if logs:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
    elif format == "json":
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)
    
    print(f"Saved training logs to {filepath}")


def get_class_name(class_idx: int) -> str:
    """
    Get class name from index.
    
    Args:
        class_idx: Class index (0, 1, or 2)
    
    Returns:
        Class name string
    """
    return REVERSE_CLASS_MAP.get(class_idx, "unknown")


def get_class_index(class_name: str) -> int:
    """
    Get class index from name.
    
    Args:
        class_name: Class name ('low', 'moderate', or 'high')
    
    Returns:
        Class index
    """
    from src.config import CLASS_NAME_MAP
    return CLASS_NAME_MAP.get(class_name.lower(), -1)


def predict_with_confidence(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cuda"
) -> Tuple[str, float, torch.Tensor]:
    """
    Get prediction and confidence from model.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: Device to run on
    
    Returns:
        Tuple of (class_name, confidence, probabilities)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        
        class_name = get_class_name(pred_idx.item())
        confidence_score = confidence.item()
    
    return class_name, confidence_score, probabilities[0].cpu()


def format_confidence(confidence: float) -> str:
    """
    Format confidence as percentage string.
    
    Args:
        confidence: Confidence value (0-1)
    
    Returns:
        Formatted string (e.g., "85.3%")
    """
    return f"{confidence * 100:.1f}%"



