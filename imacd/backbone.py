from __future__ import annotations

import torch
from torch import nn
from . import config as CW


def normalization(channels: int, encoder_norm: str | None = None) -> nn.Module:
    norm = encoder_norm or CW.ENCODER_NORM
    if norm == "batch":
        return nn.BatchNorm1d(channels)
    if norm != "group":
        raise ValueError(f"Unknown encoder normalization: {norm}")
    if channels % 8:
        raise ValueError("GroupNorm requires channels divisible by 8")
    return nn.GroupNorm(8, channels)


class ConvEncoder1D(nn.Module):
    """Compact 1D encoder: [B,1,L] -> [B,D]."""

    def __init__(self, embed_dim: int = 256, channels: int = 64,
                 encoder_norm: str | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3),
            normalization(channels, encoder_norm),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2),
            normalization(channels * 2, encoder_norm),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            normalization(channels * 4, encoder_norm),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
