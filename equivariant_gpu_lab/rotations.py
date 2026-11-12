"""Pole-safe ZYZ Euler angles and algebraic Chebyshev harmonic recurrence."""

from __future__ import annotations

from typing import Tuple

import torch


def vectors_to_zyz_angles(vectors: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract pole-safe ``(beta, alpha)`` spherical angles aligning +y with ``vectors``.

    Promotes BF16/FP16 inputs to FP32 internally and uses ``atan2(r_xz, y)`` to avoid
    ``acos`` gradient singularities at ``y = +/- 1``.
    """
    orig_dtype = vectors.dtype
    work = vectors.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else vectors
    vx, vy, vz = work[..., 0], work[..., 1], work[..., 2]

    xz_sq = vx * vx + vz * vz
    xz_zero = xz_sq <= eps * eps
    safe_vx = torch.where(xz_zero, torch.zeros_like(vx), vx)
    safe_vz = torch.where(xz_zero, torch.ones_like(vz), vz)
    r_xz = torch.where(xz_zero, torch.zeros_like(xz_sq), torch.sqrt(torch.where(xz_zero, torch.ones_like(xz_sq), xz_sq)))

    all_zero = xz_zero & (vy.abs() <= eps)
    safe_vy = torch.where(all_zero, torch.ones_like(vy), vy)

    beta = torch.atan2(r_xz, safe_vy)
    alpha = torch.where(xz_zero, torch.zeros_like(vx), torch.atan2(safe_vx, safe_vz))
    return beta.to(orig_dtype), alpha.to(orig_dtype)


def chebyshev_trig_ladder(theta: torch.Tensor, lmax: int) -> torch.Tensor:
    """Compute ``[cos(l*theta), ..., cos(theta), 1, sin(theta), ..., sin(l*theta)]`` algebraically."""
    if lmax < 0:
        raise ValueError("lmax must be >= 0")
    c1 = torch.cos(theta)
    s1 = torch.sin(theta)
    ones = torch.ones_like(theta)
    if lmax == 0:
        return ones.unsqueeze(-1)

    cos_terms = [ones, c1]
    sin_terms = [torch.zeros_like(theta), s1]
    for m in range(2, lmax + 1):
        cm = cos_terms[m - 1] * c1 - sin_terms[m - 1] * s1
        sm = sin_terms[m - 1] * c1 + cos_terms[m - 1] * s1
        cos_terms.append(cm)
        sin_terms.append(sm)

    high_cos = [cos_terms[m] for m in range(lmax, 0, -1)]
    high_sin = [sin_terms[m] for m in range(1, lmax + 1)]
    return torch.stack(high_cos + [ones] + high_sin, dim=-1)
