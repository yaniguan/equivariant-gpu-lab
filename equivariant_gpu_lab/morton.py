"""Vectorized 63-bit 3D Morton (Z-order) space-filling curve encoding."""

from __future__ import annotations

import torch


def _part1by2_21bit(n: torch.Tensor) -> torch.Tensor:
    n = n.to(torch.int64) & 0x1FFFFF
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


def morton_encode_3d(positions: torch.Tensor, bits: int = 21) -> torch.Tensor:
    """Map 3D atomic coordinates ``(N, 3)`` to 63-bit Morton keys ``(N,)``."""
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions of shape (N, 3), got {tuple(positions.shape)}")
    if not (1 <= bits <= 21):
        raise ValueError("bits must be between 1 and 21")
    if positions.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int64, device=positions.device)

    pos = positions.detach().to(torch.float64)
    mins = pos.min(dim=0).values
    span = (pos.max(dim=0).values - mins).clamp_min(1e-12)
    scale = float((1 << bits) - 1)
    q = torch.round(((pos - mins) / span) * scale).to(torch.int64).clamp(0, (1 << bits) - 1)
    mx = _part1by2_21bit(q[:, 0])
    my = _part1by2_21bit(q[:, 1]) << 1
    mz = _part1by2_21bit(q[:, 2]) << 2
    return mx | my | mz
