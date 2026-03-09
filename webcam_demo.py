"""
Real-time webcam demo for Facial Stress Indicator.

Opens webcam, detects faces, and shows stress level predictions in real-time.

Usage:
    python -m src.webcam_demo
    python -m src.webcam_demo --checkpoint outputs/checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

from src.config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES, WEBCAM_INDEX
from models.mobilenet_stress import load_model
from src.transforms import get_inference_transforms, crop_eyes_region, numpy_to_pil
from src.utils import predict_with_confidence, format_confidence


def draw_prediction(
    frame: np.ndarray,
    class_name: str,
    confidence: float,
    bbox: tuple = None
) -> np.ndarray:
    """
    Draw prediction on frame.
    
    Args:
        frame: Input frame
        class_name: Predicted class name
        confidence: Confidence score
        bbox: Optional bounding box (x, y, w, h) for face
    
    Returns:
        Frame with prediction drawn
    """
    # Color mapping
    color_map = {
        'low': (0, 255, 0),      # Green
        'moderate': (0, 165, 255),  # Orange
        'high': (0, 0, 255)      # Red
    }
    
    color = color_map.get(class_name.lower(), (255, 255, 255))
    
    # Draw bounding box if provided
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Prepare text
    label_text = f"{class_name.upper()}: {format_confidence(confidence)}"
    
    # Draw background rectangle for text
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    if bbox:
        text_x = bbox[0]
        text_y = bbox[1] - 10 if bbox[1] > 30 else bbox[1] + text_height + 10
    else:
        text_x = 10
        text_y = 30
    
    # Background rectangle
    cv2.rectangle(
        frame,
        (text_x - 5, text_y - text_height - 5),
        (text_x + text_width + 5, text_y + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    
    return frame


def preprocess_frame(
    frame: np.ndarray,
    crop_eyes: bool = False,
    transform=None
) -> torch.Tensor:
    """
    Preprocess frame for inference.
    
    Args:
        frame: Input frame (BGR format)
        crop_eyes: Whether to crop periocular region
        transform: Transform to apply
    
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    if transform is None:
        transform = get_inference_transforms()
    
    if crop_eyes:
        # Crop eyes region
        frame_cropped = crop_eyes_region(frame)
    else:
        frame_cropped = frame
    
    # Convert to PIL Image
    img_pil = numpy_to_pil(frame_cropped)
    
    # Apply transforms
    img_tensor = transform(img_pil).unsqueeze(0)
    
    return img_tensor


def detect_face(frame: np.ndarray) -> tuple:
    """
    Detect face in frame using Haar cascade.
    
    Returns:
        Tuple of (x, y, w, h) or None if no face detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Return the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        return tuple(faces[0])
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Webcam demo for Facial Stress Indicator')
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
    parser.add_argument(
        '--camera',
        type=int,
        default=WEBCAM_INDEX,
        help='Camera index'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='FPS for inference (lower = less frequent predictions)'
    )
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print("=" * 60)
    print("Facial Stress Indicator - Webcam Demo")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Crop eyes: {args.crop_eyes}")
    print("\nPress 'q' to quit")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(str(checkpoint_path), device=device)
    print("Model loaded!")
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nStarting webcam...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    last_prediction = None
    last_confidence = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run inference every N frames (to improve performance)
            if frame_count % args.fps == 0:
                try:
                    # Preprocess frame
                    frame_tensor = preprocess_frame(frame, crop_eyes=args.crop_eyes)
                    
                    # Predict
                    class_name, confidence, _ = predict_with_confidence(
                        model, frame_tensor, device
                    )
                    
                    last_prediction = class_name
                    last_confidence = confidence
                except Exception as e:
                    print(f"Error during inference: {e}")
            
            # Detect face
            face_bbox = detect_face(frame)
            
            # Draw prediction
            if last_prediction:
                frame = draw_prediction(
                    frame,
                    last_prediction,
                    last_confidence,
                    bbox=face_bbox
                )
            
            # Show frame
            cv2.imshow('Facial Stress Indicator - Press Q to quit', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam demo closed.")


if __name__ == "__main__":
    main()



