# -*- coding: utf-8 -*-
"""Lightweight Skala-style model for multi-scale gradient prediction."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor(array) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    return torch.as_tensor(array, dtype=torch.float32)


class ScaleInvariantBlock(nn.Module):
    """Multi-kernel depthwise convolution followed by pointwise mixing."""

    def __init__(self, channels: int, kernel_sizes: Iterable[int] = (3, 5, 9, 17)) -> None:
        super().__init__()
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
                for k in kernel_sizes
            ]
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.depthwise]
        stacked = torch.stack(feats, dim=0).mean(dim=0)
        out = self.pointwise(stacked)
        out = self.norm(out)
        return F.relu(out)


class PersonalStretch(nn.Module):
    """Personal latent rescaling of feature channels."""

    def __init__(self, channels: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.affine = nn.Linear(latent_dim, channels)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.affine(latent)).unsqueeze(-1)  # [B, C, 1]
        return x * (0.5 + scale)


class UFieldEncoder(nn.Module):
    """Universal field encoder extracting scale-invariant features."""

    def __init__(
        self,
        in_ch: int = 9,
        hidden: int = 32,
        depth: int = 3,
        kernel_sizes: Sequence[int] = (3, 5, 9, 17),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList(ScaleInvariantBlock(hidden, kernel_sizes) for _ in range(depth))
        self.output_proj = nn.Conv1d(hidden, hidden, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # input: [B, T, C] -> Conv1d expects [B, C, T]
        hidden = sequence.transpose(1, 2)
        hidden = F.relu(self.input_proj(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_proj(hidden)
        return hidden.transpose(1, 2)


class SkalaHead(nn.Module):
    """Combine universal representation and personal latent to predict gradients."""

    def __init__(self, hidden: int = 32, out_ch: int = 9, latent_dim: int = 8) -> None:
        super().__init__()
        self.stretch = PersonalStretch(hidden, latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_ch),
        )

    def forward(self, universal: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        hidden = universal.transpose(1, 2)
        hidden = self.stretch(hidden, latent)
        hidden = hidden.transpose(1, 2)
        return self.proj(hidden)


class SkalaModel(nn.Module):
    """Full Skala model mapping sequences to gradient predictions."""

    def __init__(
        self,
        in_ch: int = 9,
        hidden: int = 32,
        depth: int = 3,
        latent_dim: int = 8,
        kernel_sizes: Sequence[int] = (3, 5, 9, 17),
    ) -> None:
        super().__init__()
        self.encoder = UFieldEncoder(in_ch, hidden, depth, kernel_sizes)
        self.head = SkalaHead(hidden, in_ch, latent_dim)

    def forward(self, sequence: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence)
        return self.head(encoded, latent)


def structure_preserving_loss(
    model: nn.Module,
    sequence: torch.Tensor,
    latent: torch.Tensor,
    scales: Sequence[float] = (0.9, 1.1),
) -> torch.Tensor:
    """Encourage outputs to remain consistent under temporal rescaling."""

    def _resample(input_seq: torch.Tensor, scale: float) -> torch.Tensor:
        batch, time, channels = input_seq.shape
        new_time = max(2, int(time * scale))
        target_grid = torch.linspace(0, 1, new_time, device=input_seq.device).unsqueeze(0).repeat(batch, 1)
        source_grid = torch.linspace(0, 1, time, device=input_seq.device)
        indices = torch.searchsorted(source_grid, target_grid)
        indices = torch.clamp(indices, 1, time - 1)
        left = source_grid[indices - 1]
        right = source_grid[indices]
        weight = (target_grid - left) / (right - left + 1e-8)
        left_val = input_seq[:, indices - 1, :]
        right_val = input_seq[:, indices, :]
        return (1 - weight.unsqueeze(-1)) * left_val + weight.unsqueeze(-1) * right_val

    with torch.no_grad():
        baseline = model(sequence, latent)

    total = 0.0
    for scale in scales:
        scaled = _resample(sequence, scale)
        predicted = model(scaled, latent)
        length = min(baseline.shape[1], predicted.shape[1])
        total += F.mse_loss(baseline[:, :length, :], predicted[:, :length, :])

    return total / max(len(scales), 1)


def skala_predict_gradient(
    sequence_np,
    latent_np=None,
    win: int = 7,
    kernel_sizes: Sequence[int] = (3, 5, 9, 17),
) -> torch.Tensor:
    """Predict terminal gradient vector from recent window."""

    window = _to_tensor(sequence_np[-win:]).unsqueeze(0)
    latent = torch.zeros(1, 8) if latent_np is None else _to_tensor(latent_np).view(1, -1)

    model = SkalaModel(
        in_ch=window.shape[-1],
        hidden=32,
        depth=2,
        latent_dim=latent.shape[1],
        kernel_sizes=kernel_sizes,
    )
    model.eval()
    with torch.no_grad():
        output = model(window, latent)
    return output[0, -1, :].cpu().numpy()
