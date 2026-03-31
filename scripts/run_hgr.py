#!/usr/bin/env python3
"""Hand Gesture Recognition - Main Application.

Controls mouse and keyboard using hand gestures detected via webcam.

Gesture mappings:
    palm    → switch window (Alt+Tab)
    fist    → Tab key
    like    → Enter key
    dislike → Backspace key
    ok      → Space key
    peace   → close window (Alt+F4)
    one     → open browser (google.com)

Controls:
    q       → quit
    p       → pause/resume actions
    r       → reset gesture state

Safety:
    - Move mouse to screen corner to abort (pyautogui failsafe)
    - Press 'p' to pause actions while keeping preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None

import torch
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from hand_gesture_control.model import load_checkpoint
from hand_gesture_control.inference import PredictionSmoother, GestureState
from hand_gesture_control.actions import ActionMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition - Control your computer with gestures"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/hagrid_efficientnet.pt"),
        help="Model checkpoint path",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=10,
        help="Frames to hold gesture before triggering (~0.33s at 30fps)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=20,
        help="Frames before same gesture can trigger again",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable actions (preview only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable action logging to console",
    )
    return parser.parse_args()


def _landmarks_to_bbox(
    landmarks, width: int, height: int, margin: float = 0.15
) -> tuple[int, int, int, int] | None:
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    if not xs or not ys:
        return None

    x_min = max(0.0, min(xs) - margin)
    y_min = max(0.0, min(ys) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_max = min(1.0, max(ys) + margin)

    left = int(x_min * width)
    top = int(y_min * height)
    right = int(x_max * width)
    bottom = int(y_max * height)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def main() -> None:
    args = parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device)
    print(f"Loaded. Classes: {list(meta.class_to_idx.keys())}")

    # Initialize components
    num_classes = len(meta.class_to_idx)
    smoother = PredictionSmoother(
        num_classes=num_classes,
        window_size=5,
        ema_alpha=0.4,
        confidence_threshold=args.confidence,
    )
    gesture_state = GestureState(
        min_hold_frames=args.hold_frames,
        confidence_threshold=args.confidence,
    )
    action_mapper = ActionMapper(
        enabled=not args.no_actions,
        verbose=not args.quiet,
        cooldown_duration=args.cooldown,
    )

    # Set up image transforms
    weights = models.EfficientNet_B0_Weights.DEFAULT
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((meta.image_size, meta.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Initialize MediaPipe
    hands = None
    mp_draw = None
    mp_hands = None
    if mp is not None:
        try:
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"MediaPipe init failed: {e}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    print("\n" + "=" * 50)
    print("Hand Gesture Recognition Active")
    print("=" * 50)
    print("Controls: q=quit, p=pause actions, r=reset")
    print("Safety: Move mouse to corner to abort")
    print("=" * 50 + "\n")

    paused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Defaults
        label = "no_hand"
        confidence = 0.0
        bbox = None
        crop = frame_rgb
        stable_gesture = "none"
        just_triggered = False
        action_taken = None

        # Detect hand
        if hands is not None:
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                bbox = _landmarks_to_bbox(hand_landmarks, w, h)
                if bbox:
                    left, top, right, bottom = bbox
                    crop = frame_rgb[top:bottom, left:right]
                if mp_draw is not None:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Classify gesture
        if crop.size > 0:
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]

                # Smooth prediction
                idx, confidence = smoother.update(probs)
                label = meta.idx_to_class[idx]

                # Check for stable gesture
                stable_gesture, just_triggered = gesture_state.update(label, confidence)

                # Execute action
                if not paused and just_triggered:
                    action_taken = action_mapper.execute(stable_gesture, just_triggered)

        # Update cooldown
        action_mapper.tick()

        # Draw bounding box
        if bbox:
            left, top, right, bottom = bbox
            color = (0, 255, 0) if just_triggered else (0, 180, 0)
            thickness = 3 if just_triggered else 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        # Draw status
        # Line 1: Current prediction
        text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Line 2: Stable gesture
        if stable_gesture != "none":
            color = (0, 255, 255) if just_triggered else (0, 255, 0)
            cv2.putText(frame, f"ACTIVE: {stable_gesture}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Hold gesture...", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

        # Line 3: Action taken
        if action_taken:
            cv2.putText(frame, f"ACTION: {action_taken}", (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Line 4: Status bar
        status = "PAUSED" if paused else ("ACTIONS OFF" if args.no_actions else "READY")
        status_color = (0, 0, 255) if paused else ((128, 128, 128) if args.no_actions else (0, 255, 0))
        cv2.putText(frame, status, (w - 150, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Show frame
        cv2.imshow("Hand Gesture Control", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"Actions {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("r"):
            smoother.reset()
            gesture_state.reset()
            action_mapper.reset()
            print("State reset")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
