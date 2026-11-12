"""Dual-layout O(3)-equivariant RMSNorm and LayerNorm with FP32 variance accumulation."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class EquivariantRMSNorm(nn.Module):
    """Equivariant RMSNorm across ``(mul, 2l+1)`` or ``(2l+1, mul)`` irrep blocks."""

    def __init__(
        self,
        spec: Sequence[Tuple[int, int, bool]],
        layout: str = "mul_ir",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError(f"Unsupported layout: {layout}")
        self.spec = list(spec)
        self.layout = layout
        self.eps = eps
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(mul)) for mul, _, _ in self.spec])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        work = x.to(torch.float32)
        batch_shape = work.shape[:-1]
        outputs = []
        offset = 0
        for idx, (mul, dim, _is_scalar) in enumerate(self.spec):
            seg_size = mul * dim
            seg = work[..., offset : offset + seg_size]
            if self.layout == "mul_ir":
                blk = seg.reshape(*batch_shape, mul, dim)
                ms = (blk * blk).sum(dim=-1).mean(dim=-1, keepdim=True)
                scale = torch.rsqrt(ms + self.eps).unsqueeze(-1)
                w = self.weights[idx].to(torch.float32).view(*([1] * len(batch_shape)), mul, 1)
                out_blk = blk * scale * w
            else:
                blk = seg.reshape(*batch_shape, dim, mul)
                ms = (blk * blk).sum(dim=-2).mean(dim=-1, keepdim=True)
                scale = torch.rsqrt(ms + self.eps).unsqueeze(-2)
                w = self.weights[idx].to(torch.float32).view(*([1] * len(batch_shape)), 1, mul)
                out_blk = blk * scale * w
            outputs.append(out_blk.reshape(*batch_shape, seg_size))
            offset += seg_size
        return torch.cat(outputs, dim=-1).to(orig_dtype)
