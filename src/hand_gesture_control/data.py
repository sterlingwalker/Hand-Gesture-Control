from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Some HaGrid images can be truncated; allow loading to avoid DataLoader crashes.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: dict[str, int]


def _build_normalize_transform() -> transforms.Normalize:
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
    return transforms.Normalize(mean=mean, std=std)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = _build_normalize_transform()
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoaders:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(image_size, False))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transforms(image_size, False))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return DataLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        class_to_idx=train_dataset.class_to_idx,
    )


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.is_file() and is_image_file(path)]
