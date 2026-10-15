"""I implement a compact 3D CNN backbone for fixed-shape MRI tensors."""

from __future__ import annotations

import torch
from torch import nn


class Small3DBackbone(nn.Module):
    """I encode ``[batch, channel, depth, row, column]`` arrays into feature vectors."""

    def __init__(self, in_channels: int = 3, base_channels: int = 8) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("in_channels and base_channels must be positive")
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output_features = base_channels * 4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"I expect a 5D MRI batch tensor, got shape {tuple(image.shape)}")
        return self.features(image).flatten(start_dim=1)
