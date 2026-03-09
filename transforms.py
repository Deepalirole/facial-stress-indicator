"""
Data transforms and preprocessing functions for Facial Stress Indicator.

Includes training augmentations and validation/test transforms.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, Optional

from src.config import IMG_SIZE, MEAN, STD


def get_train_transforms():
    """
    Get training transforms with data augmentation.
    
    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_val_transforms():
    """
    Get validation/test transforms (no augmentation).
    
    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_inference_transforms():
    """
    Get transforms for inference (same as validation).
    
    Returns:
        torchvision.transforms.Compose: Inference transforms
    """
    return get_val_transforms()


def crop_eyes_region(image: np.ndarray, face_cascade_path: Optional[str] = None) -> np.ndarray:
    """
    Detect face and crop periocular region (around eyes).
    
    This is a stub implementation. For production, you may want to:
    - Use more robust face detection (MTCNN, dlib, etc.)
    - Better eye region localization
    - Handle edge cases (no face detected, multiple faces, etc.)
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        face_cascade_path: Path to Haar cascade XML file (optional)
    
    Returns:
        Cropped periocular region as numpy array, or original image if face not detected
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    if face_cascade_path is None:
        # Use default OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    else:
        cascade_path = face_cascade_path
    
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the first detected face
            x, y, w, h = faces[0]
            
            # Crop periocular region: upper-middle part of face
            # Adjust these ratios based on your needs
            eye_y_start = int(y + h * 0.15)  # Start a bit below top of face
            eye_y_end = int(y + h * 0.65)    # End at about 2/3 down the face
            eye_x_start = int(x + w * 0.1)   # Slight margin from sides
            eye_x_end = int(x + w * 0.9)
            
            cropped = image[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
            
            if cropped.size > 0:
                return cropped
    except Exception as e:
        print(f"Warning: Face detection failed: {e}")
    
    # Fallback: return center crop of original image
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    crop_size = min(h, w) // 2
    y_start = max(0, center_y - crop_size // 2)
    y_end = min(h, center_y + crop_size // 2)
    x_start = max(0, center_x - crop_size)
    x_end = min(w, center_x + crop_size)
    
    return image[y_start:y_end, x_start:x_end]


def numpy_to_pil(image: np.ndarray) -> 'PIL.Image':
    """
    Convert numpy array (BGR) to PIL Image (RGB).
    
    Args:
        image: numpy array in BGR format
    
    Returns:
        PIL Image in RGB format
    """
    from PIL import Image
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)



