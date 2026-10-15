"""I map compact 3D MRI features to twelve finding logits."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import Small3DBackbone


class BaselineClassifier(nn.Module):
    """I provide a small 3D CNN classifier with exactly twelve raw logits."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 8,
        output_count: int = 12,
    ) -> None:
        super().__init__()
        if output_count != 12:
            raise ValueError("The KneeScope-12 baseline must output exactly 12 logits")
        self.backbone = Small3DBackbone(in_channels, base_channels)
        self.classifier = nn.Linear(self.backbone.output_features, output_count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(image))
