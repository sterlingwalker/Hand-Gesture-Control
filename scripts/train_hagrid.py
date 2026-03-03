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
from hand_gesture_control.model import build_model, save_checkpoint
from hand_gesture_control.train_utils import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EfficientNet gesture classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/hagrid"),
        help="ImageFolder dataset directory.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hagrid_efficientnet.pt"),
        help="Checkpoint output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(loaders.class_to_idx), freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders.train, optimizer, criterion, device)
        val_metrics = evaluate(model, loaders.val, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f}"
        )

        if val_metrics.accuracy > best_val:
            best_val = val_metrics.accuracy
            save_checkpoint(args.output, model, loaders.class_to_idx, args.image_size)
            print(f"Saved checkpoint to {args.output} (val_acc={best_val:.3f})")

    print(f"Best val accuracy: {best_val:.3f}")


if __name__ == "__main__":
    main()
