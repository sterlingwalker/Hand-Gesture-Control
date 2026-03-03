from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from hand_gesture_control.data import is_image_file

SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small HaGrid subset in ImageFolder format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/hagrid"),
        help="Folder containing HaGrid images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/hagrid"),
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated class list. Defaults to all detected classes.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Max images per class (after filtering).",
    )
    parser.add_argument("--val", type=float, default=0.1, help="Validation split.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--link-type",
        type=str,
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place files in the processed dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory.",
    )
    return parser.parse_args()


def _extract_class_from_path(raw_dir: Path, image_path: Path) -> str | None:
    rel_parts = image_path.relative_to(raw_dir).parts
    if not rel_parts:
        return None
    first = rel_parts[0].lower()
    if first in SPLIT_NAMES:
        if len(rel_parts) < 2:
            return None
        return rel_parts[1]
    return rel_parts[0]


def _collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    class_to_images: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*"):
        if not path.is_file() or not is_image_file(path):
            continue
        class_name = _extract_class_from_path(raw_dir, path)
        if not class_name:
            continue
        class_to_images.setdefault(class_name, []).append(path)
    return class_to_images


def _prepare_output_dirs(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory {out_dir} exists. Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)


def _place_file(src: Path, dst: Path, link_type: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_type == "copy":
        shutil.copy2(src, dst)
        return
    if dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    if not raw_dir.exists():
        raise SystemExit(f"Raw directory {raw_dir} does not exist.")

    class_to_images = _collect_images(raw_dir)
    if not class_to_images:
        raise SystemExit(f"No images found under {raw_dir}.")

    if args.classes:
        requested = [name.strip() for name in args.classes.split(",") if name.strip()]
        class_to_images = {k: v for k, v in class_to_images.items() if k in requested}

    if not class_to_images:
        raise SystemExit("No classes matched the provided filters.")

    random.seed(args.seed)
    _prepare_output_dirs(args.out_dir, args.overwrite)

    for class_name, images in sorted(class_to_images.items()):
        random.shuffle(images)
        if args.max_per_class:
            images = images[: args.max_per_class]
        total = len(images)
        if total == 0:
            continue

        test_count = max(1, int(total * args.test))
        val_count = max(1, int(total * args.val))
        train_count = max(1, total - test_count - val_count)
        if train_count <= 0:
            train_count = max(1, total - test_count)
            val_count = total - test_count - train_count

        splits = {
            "train": images[:train_count],
            "val": images[train_count : train_count + val_count],
            "test": images[train_count + val_count :],
        }

        for split, split_images in splits.items():
            for img in split_images:
                dst = args.out_dir / split / class_name / img.name
                _place_file(img, dst, args.link_type)

        print(
            f"{class_name}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test"
        )

    print(f"Prepared dataset at {args.out_dir}")


if __name__ == "__main__":
    main()
