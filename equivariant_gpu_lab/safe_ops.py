"""Zero-safe Euclidean norms, unit vectors, and masked Verlet padding helpers."""

from __future__ import annotations

from typing import Optional

import torch


def safe_norm(vectors: torch.Tensor, eps: float = 1e-12, keepdim: bool = False) -> torch.Tensor:
    """Double-where Euclidean norm with finite 1st and 2nd derivatives at r = 0."""
    sq = (vectors * vectors).sum(dim=-1, keepdim=keepdim)
    is_zero = sq <= eps * eps
    safe_sq = torch.where(is_zero, torch.ones_like(sq), sq)
    return torch.where(is_zero, torch.zeros_like(sq), torch.sqrt(safe_sq))


def safe_normalize(
    vectors: torch.Tensor,
    eps: float = 1e-12,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize 3D vectors to unit length while mapping zero/masked slots smoothly to 0."""
    sq = (vectors * vectors).sum(dim=-1, keepdim=True)
    is_zero = sq <= eps * eps
    if mask is not None:
        is_zero = is_zero | (~mask.to(torch.bool).unsqueeze(-1))
    safe_vecs = torch.where(is_zero, torch.zeros_like(vectors), vectors)
    safe_sq = torch.where(is_zero, torch.ones_like(sq), (safe_vecs * safe_vecs).sum(dim=-1, keepdim=True))
    unit = safe_vecs * torch.rsqrt(safe_sq)
    return torch.where(is_zero, torch.zeros_like(unit), unit)
