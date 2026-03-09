# Facial Stress Indicator – Eye Fatigue & Dark Circles Detector

A deep learning project that detects stress and fatigue levels from facial/periocular images using a CNN-based classifier. The model classifies images into three categories: **Low stress**, **Moderate stress**, and **High stress** based on visual cues like dark circles, eye bags, and tired appearance.

## 🎯 Project Overview

This project implements a complete end-to-end pipeline for:
- **Manual image labeling** (for unlabeled datasets)
- **Model training** (MobileNetV2-based CNN)
- **Model evaluation** (metrics, confusion matrix)
- **Inference** (single image prediction)
- **Real-time webcam demo** (live stress detection)

**Note**: This is not a medical diagnostic tool, just a preliminary stress/fatigue awareness system.

## 📁 Project Structure

```
facial_stress_indicator/
├── data/
│   ├── raw/
│   │   └── all/              # Place your unlabeled images here
│   ├── labeled/              # Auto-created during labeling
│   │   ├── low/
│   │   ├── moderate/
│   │   └── high/
│   └── labels.csv            # CSV file with labels
├── models/
│   ├── __init__.py
│   └── mobilenet_stress.py   # MobileNetV2 model
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── dataset.py            # Dataset and DataLoader utilities
│   ├── transforms.py         # Data preprocessing and augmentation
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation script
│   ├── infer.py              # Inference script
│   ├── webcam_demo.py        # Real-time webcam demo
│   └── utils.py              # Utility functions
├── tools/
│   └── labeling_tool.py      # Manual labeling tool
├── outputs/
│   ├── checkpoints/          # Saved models
│   ├── logs/                 # Training logs
│   ├── metrics/              # Evaluation metrics
│   ├── confusion_matrices/   # Confusion matrix plots
│   └── sample_predictions/   # Sample predictions
├── app/
│   └── app_streamlit.py      # Optional Streamlit UI
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your unlabeled images in `data/raw/all/`:

```bash
data/raw/all/
  ├── img_00001.jpg
  ├── img_00002.jpg
  └── ...
```

### 3. Label Your Images

Run the labeling tool to manually label your images:

```bash
python tools/labeling_tool.py
```

**Controls:**
- Press `1` → Label as **Low stress** / No visible fatigue
- Press `2` → Label as **Moderate stress** / Mild fatigue
- Press `3` → Label as **High stress** / Strong visible fatigue
- Press `s` → Skip this image
- Press `q` → Quit and save progress

The tool will:
- Save labels to `data/labels.csv`
- Copy images to `data/labeled/<class>/` folders
- Resume from where you left off on next run

### 4. Train the Model

```bash
python -m src.train
```

**Options:**
```bash
python -m src.train --epochs 50 --batch-size 32 --lr 0.0002 --device cuda
```

Training will:
- Create train/val/test splits (70/15/15)
- Save best model to `outputs/checkpoints/best_model.pth`
- Save training logs to `outputs/logs/`

### 5. Evaluate the Model

```bash
python -m src.eval --checkpoint outputs/checkpoints/best_model.pth
```

This generates:
- Overall and per-class accuracy
- Precision, recall, F1-score
- Confusion matrix plot
- Classification report

### 6. Run Inference on an Image

```bash
python -m src.infer --image path/to/image.jpg
```

**Options:**
```bash
python -m src.infer --image image.jpg --checkpoint outputs/checkpoints/best_model.pth --crop-eyes
```

### 7. Real-Time Webcam Demo

```bash
python -m src.webcam_demo
```

**Options:**
```bash
python -m src.webcam_demo --checkpoint outputs/checkpoints/best_model.pth --crop-eyes --fps 10
```

Press `q` to quit the webcam demo.

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Paths**: Data directories, output directories
- **Training**: Batch size, learning rate, epochs, etc.
- **Model**: Architecture, pretrained weights, freezing layers
- **Data**: Image size, augmentation settings
- **Device**: CUDA/CPU selection

## 📊 Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Classifier**: 3-class output (Low/Moderate/High)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=2e-4)
- **Input Size**: 224x224 RGB images

## 🔧 Dataset Formats

The project supports two labeling formats:

### Option A: CSV-based (Recommended)
- Labels stored in `data/labels.csv`
- Images in `data/raw/all/`
- Format: `filename,label`

### Option B: Folder-based
- Images organized in `data/labeled/low/`, `data/labeled/moderate/`, `data/labeled/high/`
- Uses `torchvision.datasets.ImageFolder`

The code auto-detects which format is available.

## 📈 Training Outputs

After training, check:

- **`outputs/checkpoints/best_model.pth`**: Best model (highest val accuracy)
- **`outputs/checkpoints/last_model.pth`**: Last checkpoint
- **`outputs/logs/training_logs_*.json`**: Training history
- **`outputs/metrics/metrics_*.json`**: Evaluation metrics
- **`outputs/confusion_matrices/confusion_matrix_*.png`**: Confusion matrix plots

## 🎨 Features

- ✅ Simple manual labeling tool
- ✅ Automatic train/val/test split
- ✅ Data augmentation for training
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ Comprehensive evaluation metrics
- ✅ Real-time webcam inference
- ✅ Periocular region cropping (optional)
- ✅ CPU-friendly (works without GPU)

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `src/config.py`
- Use `--device cpu` for CPU-only training

### No Images Found
- Check that images are in `data/raw/all/`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Model Not Found
- Ensure you've trained a model first: `python -m src.train`
- Or download a pretrained checkpoint

### Webcam Not Working
- Try different camera index: `--camera 1`
- Check camera permissions

## 📝 Notes

- **Periocular Cropping**: The `crop_eyes_region()` function is a basic implementation using Haar cascades. For production, consider using more robust face detection (MTCNN, dlib, etc.).
- **Performance**: For real-time inference, consider using TensorRT or ONNX for optimization.
- **Dataset Size**: Works well with 1000+ labeled images per class. More data = better performance.

## 🔮 Future Enhancements

- [ ] Streamlit/Gradio web UI
- [ ] Multi-face detection and tracking
- [ ] Temporal smoothing for video
- [ ] Export to ONNX/TensorRT
- [ ] Fine-grained stress levels (0-100 scale)
- [ ] Integration with wearable devices

## 📄 License

This project is for educational/research purposes only. Not intended for medical diagnosis.

## 🤝 Contributing

Feel free to submit issues, fork, and create pull requests!

---

**Happy Training! 🚀**



