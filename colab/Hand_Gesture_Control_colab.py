# Hand-Gesture-Control Training Pipeline
# =======================================
# Copy this code into Google Colab cells
# Each section marked with "# === CELL === " should be a separate cell
#
# Pipeline Overview:
# 1. Setup - Clone repo, install dependencies, mount Drive
# 2. Data - Download HaGRID 30k from Kaggle, prepare splits
# 3. Train - Train EfficientNet-B0 classifier
# 4. Evaluate - Test accuracy and per-class metrics
# 5. Export - Save model to Google Drive for local inference
#
# Storage Strategy:
# /content/data/          ← HaGRID dataset (temp, ~3GB for 30k)
# /content/drive/MyDrive/ ← Trained model only (~20MB, persistent)


# === CELL 0: Verify GPU ===
# Run this first to make sure GPU is enabled
# If no GPU: Runtime → Change runtime type → GPU

import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU available: {gpu_name}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected!")
    print("   Go to Runtime → Change runtime type → GPU")


# === CELL 1: Clone Repository ===

REPO_URL = "https://github.com/sterlingwalker/Hand-Gesture-Control.git"
REPO_DIR = "Hand-Gesture-Control"

!git clone {REPO_URL}
%cd {REPO_DIR}
!git log --oneline -3


# === CELL 2: Install Dependencies ===

!pip install -q -U pip
!pip install -q -r requirements.txt

# Verify key imports
import torch
import torchvision
import cv2
import mediapipe as mp

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"MediaPipe: {mp.__version__}")


# === CELL 3: Mount Google Drive ===

from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

# Create output directory for models
MODELS_DIR = Path('/content/drive/MyDrive/hand-gesture-models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved to: {MODELS_DIR}")


# === CELL 4: Setup Kaggle API ===
# You need a Kaggle account and API token:
# 1. Go to kaggle.com/settings
# 2. Scroll to API section
# 3. Click "Create New Token"
# 4. Copy your username and API key when prompted

!pip install -q kaggle

import os
import json
from pathlib import Path

# Check if kaggle.json already exists (from previous session)
kaggle_dir = Path.home() / '.kaggle'
kaggle_json = kaggle_dir / 'kaggle.json'

if kaggle_json.exists():
    print("Kaggle credentials already configured")
else:
    print("Enter your Kaggle credentials (from kaggle.com/settings → API → Create New Token):\n")
    username = input("Kaggle username: ").strip()
    api_key = input("Kaggle API key: ").strip()

    # Create kaggle.json
    kaggle_dir.mkdir(exist_ok=True)
    credentials = {"username": username, "key": api_key}
    kaggle_json.write_text(json.dumps(credentials))
    os.chmod(kaggle_json, 0o600)
    print("\nKaggle credentials configured")


# === CELL 5: Download HaGRID 30k Dataset ===

from pathlib import Path

RAW_DIR = Path('/content/data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download from Kaggle (~3GB)
!kaggle datasets download -d innominate817/hagrid-sample-30k-384p -p {RAW_DIR}

# Unzip
!unzip -q {RAW_DIR}/hagrid-sample-30k-384p.zip -d {RAW_DIR}/hagrid

# Show what we got
!ls -la {RAW_DIR}/hagrid/
print(f"\nHaGRID 30k downloaded to {RAW_DIR}/hagrid/")


# === CELL 6: Explore Dataset Structure ===

from pathlib import Path
import os

hagrid_base = Path('/content/data/raw/hagrid')

print("All directories found:")
print("=" * 60)

all_dirs = []
for root, dirs, files in os.walk(hagrid_base):
    level = root.replace(str(hagrid_base), '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root)
    n_images = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    all_dirs.append((root, folder_name, n_images, level))
    if level < 4:  # Only show first 4 levels
        if n_images > 0:
            print(f"{indent}{folder_name}/ ({n_images} images)")
        else:
            print(f"{indent}{folder_name}/")

print("\n" + "=" * 60)
print("\nAll unique folder names with images:")
folders_with_images = set()
for root, name, n_images, level in all_dirs:
    if n_images > 0:
        folders_with_images.add(name)
print(sorted(folders_with_images))

print("\n" + "=" * 60)
print("\nSample of actual files:")
all_images = list(hagrid_base.rglob("*.jpg"))[:5]
for img in all_images:
    print(f"  {img.relative_to(hagrid_base)}")


# === CELL 7: Prepare Dataset (Extract 7 Target Gestures) ===

# HaGRID 30k from Kaggle uses folder names like "train_val_palm", "train_val_fist", etc.
# We create train/val/test splits directly here

from pathlib import Path
import shutil
import random

# Our target gestures
TARGET_GESTURES = ["palm", "fist", "like", "dislike", "ok", "peace", "one"]
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
MAX_PER_CLASS = 1000  # Limit per class for faster training

random.seed(SEED)

# Find the hagrid_30k directory
hagrid_base = Path('/content/data/raw/hagrid')
hagrid_30k = None

for candidate in hagrid_base.rglob('hagrid_30k'):
    if candidate.is_dir():
        hagrid_30k = candidate
        break

if hagrid_30k is None:
    for candidate in hagrid_base.rglob('*'):
        if candidate.is_dir():
            subdirs = [d.name for d in candidate.iterdir() if d.is_dir()]
            if any(s.startswith('train_val_') for s in subdirs):
                hagrid_30k = candidate
                break

if hagrid_30k is None:
    raise SystemExit("Could not find hagrid_30k directory")

print(f"Found HaGRID data in: {hagrid_30k}")

# Create output directory
out_dir = Path('/content/data/processed/hagrid')
if out_dir.exists():
    shutil.rmtree(out_dir)

for split in ['train', 'val', 'test']:
    (out_dir / split).mkdir(parents=True)

# Process each gesture
print("\nPreparing dataset splits...")
for gesture in TARGET_GESTURES:
    src_folder = hagrid_30k / f"train_val_{gesture}"
    if not src_folder.exists():
        print(f"  {gesture}: NOT FOUND")
        continue

    # Get all images
    images = list(src_folder.glob('*.jpg')) + list(src_folder.glob('*.png'))
    random.shuffle(images)

    # Limit per class
    if len(images) > MAX_PER_CLASS:
        images = images[:MAX_PER_CLASS]

    # Split
    n_total = len(images)
    n_test = max(1, int(n_total * TEST_RATIO))
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_test - n_val

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    # Copy files
    for split_name, split_images in splits.items():
        split_dir = out_dir / split_name / gesture
        split_dir.mkdir(parents=True, exist_ok=True)
        for img in split_images:
            shutil.copy2(img, split_dir / img.name)

    print(f"  {gesture}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

print(f"\nDataset prepared at: {out_dir}")


# === CELL 8: Verify Prepared Dataset ===

from pathlib import Path

processed_dir = Path('/content/data/processed/hagrid')

print("Prepared dataset summary:\n")
for split in ['train', 'val', 'test']:
    split_dir = processed_dir / split
    if split_dir.exists():
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        total = sum(len(list((split_dir / c).glob('*'))) for c in classes)
        print(f"{split}: {total} images across {len(classes)} classes")
        if split == 'train':
            print(f"   Classes: {', '.join(classes)}")


# === CELL 9: Train Model ===
# Training configuration:
# - Model: EfficientNet-B0 (pretrained, frozen backbone)
# - Optimizer: AdamW
# - Epochs: 15
# - Batch size: 64 (good for T4 GPU)

# Ensure models directory exists
!mkdir -p /content/Hand-Gesture-Control/models

!PYTHONPATH=/content/Hand-Gesture-Control python scripts/train_hagrid.py \
    --data-dir /content/data/processed/hagrid \
    --output /content/Hand-Gesture-Control/models/hagrid_efficientnet.pt \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.001


# === CELL 10: Evaluate on Test Set ===

!PYTHONPATH=/content/Hand-Gesture-Control python scripts/eval_hagrid.py \
    --data-dir /content/data/processed/hagrid \
    --checkpoint /content/Hand-Gesture-Control/models/hagrid_efficientnet.pt


# === CELL 11: Detailed Evaluation with Confusion Matrix ===

!pip install -q scikit-learn seaborn

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/content/Hand-Gesture-Control')

from src.hand_gesture_control.model import load_checkpoint
from src.hand_gesture_control.data import build_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, meta = load_checkpoint('/content/Hand-Gesture-Control/models/hagrid_efficientnet.pt', device)
model = model.to(device)
model.eval()

# Load test data
dataloaders = build_dataloaders(
    Path('/content/data/processed/hagrid'),
    image_size=meta.image_size,
    batch_size=64,
    num_workers=2
)
test_loader = dataloaders.test

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Get class names in index order
class_names = [meta.idx_to_class[i] for i in range(len(meta.class_to_idx))]

# Classification report
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


# === CELL 12: Save Model to Google Drive ===

import shutil
from pathlib import Path
from datetime import datetime

# Source and destination
src_model = Path('/content/Hand-Gesture-Control/models/hagrid_efficientnet.pt')
drive_dir = Path('/content/drive/MyDrive/hand-gesture-models')
drive_dir.mkdir(parents=True, exist_ok=True)

# Copy with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
dst_model = drive_dir / f'hagrid_efficientnet_{timestamp}.pt'
dst_latest = drive_dir / 'hagrid_efficientnet_latest.pt'

shutil.copy(src_model, dst_model)
shutil.copy(src_model, dst_latest)

print(f"Model saved to Google Drive:")
print(f"   {dst_model}")
print(f"   {dst_latest}")
print(f"\nDownload 'hagrid_efficientnet_latest.pt' to your local machine")
print(f"   Place it in: Hand-Gesture-Control/models/")


# === CELL 13: Download Model Directly (Alternative) ===

from google.colab import files

print("Downloading model to your computer...")
files.download('/content/Hand-Gesture-Control/models/hagrid_efficientnet.pt')


# =============================================================================
# LOCAL INFERENCE (Run on your machine, not in Colab)
# =============================================================================
#
# After downloading the model:
#
# cd Hand-Gesture-Control
# python scripts/predict_webcam.py --checkpoint models/hagrid_efficientnet.pt
#
# Press 'q' to quit the demo.
# =============================================================================


# =============================================================================
# NEXT STEPS
# =============================================================================
#
# If accuracy < 85%, scale up dataset:
# !kaggle datasets download -d innominate817/hagrid-sample-120k-384p -p /content/data/raw
#
# Gesture Mapping Reference:
# +--------------+--------------+------------------+
# | Our Gesture  | HaGRID Class | Action           |
# +--------------+--------------+------------------+
# | OPEN_PALM    | palm         | Idle             |
# | FIST         | fist         | Click            |
# | THUMBS_UP    | like         | Confirm (Enter)  |
# | THUMBS_DOWN  | dislike      | Cancel (Escape)  |
# | OK_SIGN      | ok           | Mode Switch      |
# | PEACE        | peace        | (Reserved)       |
# | POINTING     | one          | Cursor Control   |
# +--------------+--------------+------------------+
#
# Checklist:
# [ ] Model achieves >85% accuracy
# [ ] Model downloaded to local machine
# [ ] Webcam demo runs locally
# [ ] Ready for gesture state machine implementation
# =============================================================================
