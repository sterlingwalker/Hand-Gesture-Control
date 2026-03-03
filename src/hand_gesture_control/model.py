from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass
class CheckpointMeta:
    class_to_idx: dict[str, int]
    image_size: int

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {v: k for k, v in self.class_to_idx.items()}


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(path: Path, model: nn.Module, class_to_idx: dict[str, int], image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": image_size,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, CheckpointMeta]:
    checkpoint = torch.load(path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    image_size = int(checkpoint.get("image_size", 224))
    model = build_model(num_classes=len(class_to_idx), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, CheckpointMeta(class_to_idx=class_to_idx, image_size=image_size)
