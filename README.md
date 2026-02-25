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
