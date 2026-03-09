"""
Configuration file for Facial Stress Indicator project.

All hyperparameters, paths, and settings are defined here.
"""

import os
import sys
from pathlib import Path

# Try to import torch for device detection, fallback to "cpu" if not available
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ========== PATHS ==========
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "all"
LABELED_DATA_DIR = DATA_DIR / "labeled"
LABELS_CSV = DATA_DIR / "labels.csv"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CONFUSION_MATRICES_DIR = OUTPUTS_DIR / "confusion_matrices"
SAMPLE_PREDICTIONS_DIR = OUTPUTS_DIR / "sample_predictions"

# Model paths
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

# ========== DATASET CONFIG ==========
NUM_CLASSES = 3
CLASS_NAMES = ["low", "moderate", "high"]
CLASS_NAME_MAP = {
    "low": 0,
    "moderate": 1,
    "high": 2
}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_NAME_MAP.items()}

# Train/Val/Test split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Image preprocessing
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# ========== TRAINING CONFIG ==========
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
# On Windows, use 0 workers to avoid multiprocessing issues
# On Linux/Mac, use multiple workers for faster data loading
NUM_WORKERS = 0 if sys.platform == 'win32' else (4 if os.cpu_count() > 4 else 2)

# Optimizer
OPTIMIZER = "adam"
MOMENTUM = 0.9  # For SGD

# Scheduler
SCHEDULER = "step"  # or "cosine", "plateau"
STEP_SIZE = 10
GAMMA = 0.1
MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Model
MODEL_NAME = "mobilenet_v2"
PRETRAINED = True
FREEZE_BACKBONE = False  # Set to True to freeze early layers

# ========== INFERENCE CONFIG ==========
CONFIDENCE_THRESHOLD = 0.5

# ========== WEBCAM DEMO CONFIG ==========
WEBCAM_INDEX = 0
WEBCAM_FPS = 30
FACE_CASCADE_PATH = None  # Will use default OpenCV Haar cascade

# ========== LOGGING ==========
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# ========== RANDOM SEED ==========
RANDOM_SEED = 42

