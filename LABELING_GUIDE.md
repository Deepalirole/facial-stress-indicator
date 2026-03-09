# Labeling Tool Guide

## Quick Start

You have **13,233 images** ready to label in `data/raw/all/`.

### Step 1: Start the Labeling Tool

```bash
python tools/labeling_tool.py
```

### Step 2: Label Images

When the tool opens, you'll see images one by one. Use these keyboard controls:

- **Press `1`** → Label as **Low stress** / No visible fatigue
- **Press `2`** → Label as **Moderate stress** / Mild fatigue  
- **Press `3`** → Label as **High stress** / Strong visible fatigue
- **Press `s`** → Skip this image (don't label it)
- **Press `q`** → Quit and save progress

### Step 3: Progress is Saved Automatically

- Labels are saved to `data/labels.csv`
- Images are copied to `data/labeled/low/`, `data/labeled/moderate/`, or `data/labeled/high/`
- You can quit anytime with `q` and resume later - it will skip already labeled images

### Step 4: Recommended Labeling Strategy

**Start Small:**
- Label 50-100 images per class first (150-300 total)
- This is enough to train an initial model
- You can always label more later to improve the model

**Labeling Guidelines:**
- **Low (1)**: Fresh, alert appearance, no dark circles, well-rested look
- **Moderate (2)**: Some tiredness visible, mild dark circles, slightly fatigued
- **High (3)**: Strong fatigue signs, prominent dark circles, very tired appearance

### Step 5: After Labeling

Once you have labeled images, train the model:

```bash
python -m src.train
```

## Tips

- **Take breaks**: Labeling can be tiring. Label in batches of 50-100 images.
- **Be consistent**: Try to use the same criteria throughout.
- **Skip unclear images**: Press `s` if you're unsure - you can come back to them later.
- **Check progress**: The tool shows how many images you've labeled and how many remain.

## Next Steps

After labeling and training:
1. Evaluate: `python -m src.eval`
2. Test on images: `python -m src.infer --image path/to/image.jpg`
3. Use the Streamlit app (already running at http://localhost:8501)
