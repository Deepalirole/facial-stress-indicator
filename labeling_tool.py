"""
Manual Labeling Tool for Facial Stress Indicator Dataset

This tool helps you label images into 3 classes:
- 1: Low stress / No visible fatigue
- 2: Moderate stress / Mild fatigue  
- 3: High stress / Strong visible fatigue

Usage:
    python tools/labeling_tool.py

Controls:
    - Press '1' to label as Low
    - Press '2' to label as Moderate
    - Press '3' to label as High
    - Press 's' to skip (don't label this image)
    - Press 'q' to quit and save progress
"""

import os
import cv2
import csv
import glob
from pathlib import Path
from typing import Set, Optional

# ========== CONFIGURATION ==========
# Path to directory containing unlabeled images
RAW_IMAGES_DIR = "data/raw/all"

# Path to CSV file storing labels
LABELS_CSV = "data/labels.csv"

# Path to labeled directories (optional - for organizing images)
LABELED_DIR = "data/labeled"
LABELED_SUBDIRS = {
    "low": "low",
    "moderate": "moderate", 
    "high": "high"
}

# Image display settings
WINDOW_NAME = "Labeling Tool - Press 1/2/3 to label, 's' to skip, 'q' to quit"
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600

# Label mappings
LABEL_MAP = {
    ord('1'): 'low',
    ord('2'): 'moderate',
    ord('3'): 'high'
}

LABEL_NAMES = {
    'low': 'Low Stress',
    'moderate': 'Moderate Stress',
    'high': 'High Stress'
}

# ====================================


def load_existing_labels(csv_path: str) -> Set[str]:
    """Load already labeled image filenames from CSV."""
    labeled = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row['filename'])
    return labeled


def save_label(csv_path: str, filename: str, label: str, copy_to_labeled: bool = True):
    """Save label to CSV and optionally copy image to labeled directory."""
    # Append to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['filename', 'label'])
        writer.writerow([filename, label])
    
    # Optionally copy image to labeled directory
    if copy_to_labeled:
        src_path = os.path.join(RAW_IMAGES_DIR, filename)
        dst_dir = os.path.join(LABELED_DIR, LABELED_SUBDIRS[label])
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, filename)
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)


def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """Resize image for display while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img


def main():
    """Main labeling loop."""
    print("=" * 60)
    print("Facial Stress Indicator - Labeling Tool")
    print("=" * 60)
    print(f"Images directory: {RAW_IMAGES_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")
    print("\nControls:")
    print("  Press '1' → Low stress / No visible fatigue")
    print("  Press '2' → Moderate stress / Mild fatigue")
    print("  Press '3' → High stress / Strong visible fatigue")
    print("  Press 's' → Skip this image")
    print("  Press 'q' → Quit and save progress")
    print("=" * 60)
    
    # Check if raw images directory exists
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"ERROR: Directory '{RAW_IMAGES_DIR}' does not exist!")
        print(f"Please create it and add your images there.")
        return
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(RAW_IMAGES_DIR, ext)))
    
    if not image_files:
        print(f"ERROR: No images found in '{RAW_IMAGES_DIR}'!")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    # Load already labeled images
    labeled = load_existing_labels(LABELS_CSV)
    print(f"\nFound {len(image_files)} total images")
    print(f"Already labeled: {len(labeled)}")
    print(f"Remaining: {len(image_files) - len(labeled)}")
    print()
    
    # Filter out already labeled images
    unlabeled = [f for f in image_files if os.path.basename(f) not in labeled]
    
    if not unlabeled:
        print("All images are already labeled!")
        return
    
    print(f"Starting with {len(unlabeled)} unlabeled images...\n")
    
    # Create CSV file if it doesn't exist
    os.makedirs(os.path.dirname(LABELS_CSV) if os.path.dirname(LABELS_CSV) else '.', exist_ok=True)
    
    # Create labeled directories
    for subdir in LABELED_SUBDIRS.values():
        os.makedirs(os.path.join(LABELED_DIR, subdir), exist_ok=True)
    
    # Initialize OpenCV window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    labeled_count = 0
    skipped_count = 0
    
    try:
        for img_path in unlabeled:
            filename = os.path.basename(img_path)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {filename}, skipping...")
                continue
            
            # Resize for display
            display_img = resize_for_display(img.copy())
            
            # Add text overlay with instructions
            overlay = display_img.copy()
            cv2.putText(overlay, f"Image: {filename}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "1=Low | 2=Moderate | 3=High | s=Skip | q=Quit", 
                       (10, display_img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(WINDOW_NAME, overlay)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                skipped_count += 1
                print(f"Skipped: {filename}")
                continue
            elif key in LABEL_MAP:
                label = LABEL_MAP[key]
                save_label(LABELS_CSV, filename, label, copy_to_labeled=True)
                labeled_count += 1
                print(f"Labeled '{filename}' as: {LABEL_NAMES[label]} ({labeled_count} labeled, {skipped_count} skipped)")
            else:
                print(f"Invalid key. Press 1/2/3 to label, 's' to skip, or 'q' to quit.")
                # Show the same image again
                continue
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
    
    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"Labeling session complete!")
        print(f"  - Labeled: {labeled_count} images")
        print(f"  - Skipped: {skipped_count} images")
        print(f"  - Labels saved to: {LABELS_CSV}")
        print("=" * 60)


if __name__ == "__main__":
    main()



