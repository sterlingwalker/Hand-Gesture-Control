# Hand-Gesture-Control

Real-time vision-based hand gesture recognition that maps gestures to UI actions (mouse/keyboard) for touchless Human–Computer Interaction (HCI).

## Goals
- Detect hand + classify a small gesture set reliably in real time
- Map gestures to UI actions (click, scroll, swipe, zoom)
- Deliver a working demo + evaluation metrics (accuracy, latency, robustness)

## Proposed Architecture
**Pipeline A - Fastest to demo:**
1) Webcam frames → 2) **MediaPipe Hands** landmarks → 3) Temporal model (LSTM/Transformer) → 4) Gesture → 5) UI action

**Pipeline B - More CV heavy:**
1) Webcam frames → 2) Hand detector (YOLO) → 3) Crop hand ROI → 4) Classifier (CNN/3D CNN/Transformer) → 5) UI action

We’ll start with Pipeline A for speed, then optionally add YOLO to improve robustness.

## Starting Gesture Set 
 6–10 gestures that feel natural and are easy to disambiguate:
- open_palm = idle/stop
- point = move cursor (optional)
- pinch = click/drag (optional)
- swipe_left / swipe_right = back/next
- swipe_up / swipe_down = scroll
- thumbs_up = confirm
- ok_sign = mode switch

## Gesture Recognition Pipeline (HaGrid subset)
This project now includes a full image-based pipeline using a subset of HaGrid:

1) **Prepare a local HaGrid subset** in ImageFolder format
2) **Train** an EfficientNet classifier
3) **Evaluate** on a held-out test split
4) **Run realtime webcam inference** with MediaPipe hand detection

### 1) Download a HaGrid subset
Place images under `data/raw/hagrid`. The prep script can handle two common layouts:

```
data/raw/hagrid/<gesture>/<image files>
```

or

```
data/raw/hagrid/<split>/<gesture>/<image files>
```

You can keep it small (e.g., a few hundred to a few thousand images per gesture).

### 2) Prepare the dataset
This builds `data/processed/hagrid` in ImageFolder format with train/val/test splits.

```bash
python scripts/prepare_hagrid_subset.py \
  --raw-dir data/raw/hagrid \
  --out-dir data/processed/hagrid \
  --max-per-class 1000 \
  --val 0.1 \
  --test 0.1 \
  --link-type symlink \
  --overwrite
```

If you only want specific gesture classes, pass `--classes`:

```bash
python scripts/prepare_hagrid_subset.py --classes "palm,thumbs_up,ok"
```

### 3) Train
```bash
PYTHONPATH=. python scripts/train_hagrid.py \
  --data-dir data/processed/hagrid \
  --epochs 8 \
  --batch-size 32
```

The best checkpoint is saved to `models/hagrid_efficientnet.pt`.

### 4) Evaluate
```bash
PYTHONPATH=. python scripts/eval_hagrid.py \
  --data-dir data/processed/hagrid \
  --checkpoint models/hagrid_efficientnet.pt
```

### 5) Realtime webcam demo
```bash
PYTHONPATH=. python scripts/predict_webcam.py \
  --checkpoint models/hagrid_efficientnet.pt
```

Press `q` to quit.

## Notes
- The pipeline uses EfficientNet-B0 + MediaPipe Hands.
- For faster iteration, keep the dataset small at first, then scale once the pipeline works.
- If you want temporal modeling later (LSTM/Transformer on landmarks), we can extend this pipeline.
