from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from hand_gesture_control.data import build_dataloaders
from hand_gesture_control.model import load_checkpoint
from hand_gesture_control.train_utils import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained gesture classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/hagrid"),
        help="ImageFolder dataset directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/hagrid_efficientnet.pt"),
        help="Checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_checkpoint(args.checkpoint, device)
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=meta.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loaders.test, criterion, device)
    print(f"Test loss={metrics.loss:.4f} test_acc={metrics.accuracy:.3f}")


if __name__ == "__main__":
    main()
