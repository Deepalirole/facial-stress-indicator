"""
Dataset and DataLoader utilities for Facial Stress Indicator.

Supports both folder-based and CSV-based labeling.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List

from src.config import (
    RAW_DATA_DIR, LABELED_DATA_DIR, LABELS_CSV,
    CLASS_NAME_MAP, NUM_CLASSES, CLASS_NAMES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    BATCH_SIZE, NUM_WORKERS, RANDOM_SEED
)
from src.transforms import get_train_transforms, get_val_transforms


class TransformSubset(Dataset):
    """Wrapper to apply transforms to a Subset dataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)


class StressDatasetCSV(Dataset):
    """
    Custom Dataset for CSV-based labels.
    
    Reads labels from CSV file and images from raw directory.
    """
    
    def __init__(self, csv_path: str, images_dir: str, transform=None, class_map: dict = None):
        """
        Args:
            csv_path: Path to CSV file with 'filename' and 'label' columns
            images_dir: Directory containing images
            transform: Optional transform to apply to images
            class_map: Dictionary mapping label names to class indices
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_map = class_map or CLASS_NAME_MAP
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Validate columns
        if 'filename' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("CSV must have 'filename' and 'label' columns")
        
        # Filter out invalid labels
        valid_labels = set(self.class_map.keys())
        self.df = self.df[self.df['label'].isin(valid_labels)].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} labeled images from {csv_path}")
        print(f"Label distribution:")
        print(self.df['label'].value_counts().to_dict())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label_name = row['label']
        
        # Load image
        img_path = self.images_dir / filename
        if not img_path.exists():
            # Try in labeled directories
            for class_name in CLASS_NAMES:
                alt_path = LABELED_DATA_DIR / class_name / filename
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to class index
        label = self.class_map[label_name]
        
        return image, label


def create_dataloaders(
    data_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    img_size: int = 224,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    use_folder_structure: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders.
    
    Supports two modes:
    1. Folder-based: data_dir should contain subdirectories (low/, moderate/, high/)
    2. CSV-based: csv_path should point to labels.csv, images in data_dir or RAW_DATA_DIR
    
    Args:
        data_dir: Directory containing labeled images (folder structure) or raw images (CSV mode)
        csv_path: Path to labels.csv (for CSV-based mode)
        batch_size: Batch size for dataloaders
        img_size: Target image size (not used if transforms are provided)
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        use_folder_structure: If True, use ImageFolder; if False, use CSV-based dataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    torch.manual_seed(seed)
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Determine which mode to use
    if use_folder_structure and os.path.exists(LABELED_DATA_DIR):
        # Check if folder structure exists
        has_folders = all(
            os.path.exists(LABELED_DATA_DIR / class_name) 
            for class_name in CLASS_NAMES
        )
        
        if has_folders:
            print("Using folder-based dataset structure...")
            dataset_dir = LABELED_DATA_DIR
        else:
            use_folder_structure = False
    
    if not use_folder_structure:
        # Use CSV-based dataset
        if csv_path is None:
            csv_path = LABELS_CSV
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Labels CSV not found: {csv_path}\n"
                f"Please run the labeling tool first: python tools/labeling_tool.py"
            )
        
        print("Using CSV-based dataset...")
        
        # Create full dataset
        full_dataset = StressDatasetCSV(
            csv_path=csv_path,
            images_dir=data_dir or RAW_DATA_DIR,
            transform=None  # We'll apply transforms in splits
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(TRAIN_SPLIT * total_size)
        val_size = int(VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size
        
        # Create splits with different transforms
        train_dataset = StressDatasetCSV(
            csv_path=csv_path,
            images_dir=data_dir or RAW_DATA_DIR,
            transform=train_transform
        )
        val_dataset = StressDatasetCSV(
            csv_path=csv_path,
            images_dir=data_dir or RAW_DATA_DIR,
            transform=val_transform
        )
        test_dataset = StressDatasetCSV(
            csv_path=csv_path,
            images_dir=data_dir or RAW_DATA_DIR,
            transform=val_transform
        )
        
        # Get indices for splits (same for all)
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
        class_names = CLASS_NAMES
    
    else:
        # Use ImageFolder (folder-based)
        print("Using ImageFolder dataset structure...")
        dataset_dir = LABELED_DATA_DIR
        
        # Create full dataset
        full_dataset = datasets.ImageFolder(
            root=dataset_dir,
            transform=None
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(TRAIN_SPLIT * total_size)
        val_size = int(VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Apply transforms to subsets using the module-level TransformSubset class
        train_dataset = TransformSubset(train_dataset, train_transform)
        val_dataset = TransformSubset(val_dataset, val_transform)
        test_dataset = TransformSubset(test_dataset, val_transform)
        
        class_names = full_dataset.classes
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes: {class_names}\n")
    
    return train_loader, val_loader, test_loader, class_names



