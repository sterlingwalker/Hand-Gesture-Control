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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime gesture prediction from webcam.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/hagrid_efficientnet.pt"),
        help="Checkpoint path.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--bbox-margin", type=float, default=0.15)
    return parser.parse_args()


def _landmarks_to_bbox(
    landmarks, width: int, height: int, margin: float
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    mean = weights.meta.get("mean")
    std = weights.meta.get("std")
    if mean is None or std is None:
        preprocess = weights.transforms()
        mean = getattr(preprocess, "mean", None)
        std = getattr(preprocess, "std", None)
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((meta.image_size, meta.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    hands = None
    mp_draw = None
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
        except Exception:
            hands = None
            mp_draw = None
            print(
                "MediaPipe solutions API not available. "
                "Falling back to full-frame classification."
            )
    else:
        print(
            "MediaPipe not available. Falling back to full-frame classification."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label = "no_hand"
        confidence = 0.0
        bbox = None
        crop = frame_rgb

        if hands is not None:
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                bbox = _landmarks_to_bbox(
                    hand_landmarks, frame.shape[1], frame.shape[0], args.bbox_margin
                )
                if bbox:
                    left, top, right, bottom = bbox
                    crop = frame_rgb[top:bottom, left:right]
                if mp_draw is not None:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if crop.size > 0:
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                confidence, idx = torch.max(probs, dim=0)
                label = meta.idx_to_class[int(idx)]
                confidence = float(confidence)

        if bbox:
            left, top, right, bottom = bbox
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)

        text = f"{label} ({confidence:.2f})"
        if confidence < args.min_confidence:
            text = "low_confidence"
        cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
