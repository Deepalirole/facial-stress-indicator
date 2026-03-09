"""
MobileNetV2-based model for Facial Stress Indicator.

Uses pretrained MobileNetV2 as backbone and replaces classifier head
for 3-class stress/fatigue classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

from src.config import NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY, FREEZE_BACKBONE


class StressIndicatorModel(nn.Module):
    """
    MobileNetV2-based stress indicator model.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers (only train classifier)
        """
        super(StressIndicatorModel, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            print("Backbone layers frozen. Only training classifier head.")
    
    def forward(self, x):
        return self.backbone(x)


def get_mobilenet_stress_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = FREEZE_BACKBONE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Optional[object]]:
    """
    Create model, criterion, optimizer, and scheduler.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, criterion, optimizer, scheduler)
    """
    # Create model
    model = StressIndicatorModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Move to device
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler (optional - can be configured in config.py)
    scheduler = None  # Can add StepLR, CosineAnnealingLR, etc. if needed
    
    return model, criterion, optimizer, scheduler


def load_model(
    checkpoint_path: str,
    num_classes: int = NUM_CLASSES,
    device: str = "cuda"
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes (should match saved model)
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = StressIndicatorModel(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model



