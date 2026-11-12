"""Hopper O(3)-equivariant Euclidean-L2 FP8 block scaling vs broken amax scaling."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def equivariant_fp8_quantize(
    x: torch.Tensor,
    irrep_dims: Sequence[int],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize concatenated irreps ``(..., sum_k (2*l_k + 1))`` preserving SO(3)/O(3) equivariance.

    Uses the rotation-invariant Euclidean L2 norm of each ``(2l+1)`` block as the scale
    factor rather than the non-equivariant coordinate-wise ``amax``.
    """
    fmax = float(torch.finfo(fp8_dtype).max)
    chunks_q = []
    scales = []
    offset = 0
    for dim in irrep_dims:
        blk = x[..., offset : offset + dim].to(torch.float32)
        l2 = torch.linalg.vector_norm(blk, ord=2, dim=-1, keepdim=True)
        inv_scale = torch.where(l2 > eps, fmax / l2.clamp_min(eps), torch.ones_like(l2))
        scale = torch.where(l2 > eps, l2 / fmax, torch.ones_like(l2))
        q = (blk * inv_scale).clamp(-fmax, fmax).to(fp8_dtype)
        chunks_q.append(q)
        scales.append(scale)
        offset += dim
    return torch.cat(chunks_q, dim=-1), torch.cat(scales, dim=-1)


def equivariant_fp8_dequantize(
    q: torch.Tensor,
    scales: torch.Tensor,
    irrep_dims: Sequence[int],
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Restore quantized FP8 irrep blocks using their per-irrep L2 scales."""
    chunks = []
    offset = 0
    for idx, dim in enumerate(irrep_dims):
        blk = q[..., offset : offset + dim].to(torch.float32)
        sc = scales[..., idx : idx + 1]
        chunks.append((blk * sc).to(out_dtype))
        offset += dim
    return torch.cat(chunks, dim=-1)


def equivariant_fp8_ste(
    x: torch.Tensor,
    irrep_dims: Sequence[int],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Straight-through estimator wrapper for O(3)-equivariant FP8 quantization."""
    q, sc = equivariant_fp8_quantize(x, irrep_dims, fp8_dtype=fp8_dtype)
    dq = equivariant_fp8_dequantize(q, sc, irrep_dims, out_dtype=x.dtype)
    return x + (dq - x).detach()
