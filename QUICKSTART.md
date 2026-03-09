# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Images

Place your ~13,000 images in `data/raw/all/`:

```bash
# Example structure:
data/raw/all/
  ├── img_00001.jpg
  ├── img_00002.jpg
  └── ...
```

### 3. Label Your Images

Run the labeling tool:

```bash
python tools/labeling_tool.py
```

**Quick Tips:**
- Start with a small batch (100-200 images) to get familiar
- Press `1` for Low, `2` for Moderate, `3` for High
- Press `s` to skip unclear images
- Press `q` to quit and save (progress is saved automatically)
- The tool resumes from where you left off

### 4. Train Your Model

Once you have labeled images (at least 100-200 per class recommended):

```bash
python -m src.train
```

**Training will:**
- Automatically split data (70% train, 15% val, 15% test)
- Save best model to `outputs/checkpoints/best_model.pth`
- Show progress with loss and accuracy

**Expected Time:**
- CPU: ~2-4 hours for 50 epochs
- GPU: ~30-60 minutes for 50 epochs

### 5. Evaluate Your Model

```bash
python -m src.eval
```

This generates:
- Accuracy metrics
- Confusion matrix (saved as image)
- Per-class precision/recall/F1

### 6. Test on a Single Image

```bash
python -m src.infer --image path/to/your/image.jpg
```

### 7. Try Real-Time Webcam Demo

```bash
python -m src.webcam_demo
```

Press `q` to quit.

## Troubleshooting

### "No images found"
- Check that images are in `data/raw/all/`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### "CUDA out of memory"
- Reduce batch size: `python -m src.train --batch-size 16`
- Or use CPU: `python -m src.train --device cpu`

### "Model checkpoint not found"
- Make sure you've trained a model first
- Check that `outputs/checkpoints/best_model.pth` exists

### Labeling tool window doesn't show
- Make sure you have OpenCV installed: `pip install opencv-python`
- Try running from terminal (not IDE)

## Next Steps

1. **Label more images** - More data = better model
2. **Experiment with hyperparameters** - Edit `src/config.py`
3. **Try different models** - Modify `models/mobilenet_stress.py`
4. **Add more features** - Periocular cropping, data augmentation, etc.

## Minimum Requirements

- **CPU**: Works, but slower (2-4 hours training)
- **RAM**: 8GB+ recommended
- **Disk**: ~5GB for dataset + models
- **Python**: 3.9+

## Recommended Setup

- **GPU**: NVIDIA GPU with 4GB+ VRAM (much faster training)
- **RAM**: 16GB+
- **Disk**: 10GB+ free space

---

**Ready to start?** Begin with step 3 (labeling tool)!



