"""
Inference script for Facial Stress Indicator.

Run inference on a single image or batch of images.

Usage:
    python -m src.infer --image path/to/image.jpg
    python -m src.infer --image path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
from PIL import Image
import argparse
from pathlib import Path
import cv2
import numpy as np

from src.config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES
from models.mobilenet_stress import load_model
from src.transforms import get_inference_transforms, crop_eyes_region, numpy_to_pil
from src.utils import predict_with_confidence, format_confidence, get_class_name


def load_and_preprocess_image(
    image_path: str,
    crop_eyes: bool = False,
    transform=None
) -> torch.Tensor:
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to image file
        crop_eyes: Whether to crop periocular region
        transform: Transform to apply
    
    Returns:
        Preprocessed image tensor (1, C, H, W)
    """
    if transform is None:
        transform = get_inference_transforms()
    
    # Load image
    if crop_eyes:
        # Load with OpenCV for face detection
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Crop eyes region
        img_cropped = crop_eyes_region(img_cv)
        
        # Convert to PIL
        img_pil = numpy_to_pil(img_cropped)
    else:
        # Load directly with PIL
        img_pil = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def predict_image(
    model: nn.Module,
    image_path: str,
    device: str = "cuda",
    crop_eyes: bool = False
) -> dict:
    """
    Predict stress level for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run on
        crop_eyes: Whether to crop periocular region
    
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess
    image_tensor = load_and_preprocess_image(image_path, crop_eyes=crop_eyes)
    
    # Predict
    class_name, confidence, probabilities = predict_with_confidence(
        model, image_tensor, device
    )
    
    # Get all class probabilities
    prob_dict = {
        CLASS_NAMES[i]: float(prob.item())
        for i, prob in enumerate(probabilities)
    }
    
    return {
        'image_path': image_path,
        'predicted_class': class_name,
        'confidence': confidence,
        'confidence_percent': format_confidence(confidence),
        'all_probabilities': prob_dict
    }


def main():
    parser = argparse.ArgumentParser(description='Inference for Facial Stress Indicator')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(BEST_MODEL_PATH),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=DEVICE,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--crop-eyes',
        action='store_true',
        help='Crop periocular region before inference'
    )
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print("=" * 60)
    print("Facial Stress Indicator - Inference")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Crop eyes: {args.crop_eyes}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(str(checkpoint_path), device=device)
    
    # Predict
    print("\nRunning inference...")
    result = predict_image(model, str(image_path), device, args.crop_eyes)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nImage: {result['image_path']}")
    print(f"Predicted Class: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence_percent']}")
    print("\nAll Class Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        print(f"  {class_name.capitalize()}: {format_confidence(prob)}")
    print("=" * 60)


if __name__ == "__main__":
    main()



