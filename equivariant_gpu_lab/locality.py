"""L2-cache-friendly molecular graph reordering for IndexedLinear and message passing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .morton import morton_encode_3d


@dataclass
class ReorderedGraph:
    perm: torch.Tensor
    inv_perm: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    species: Optional[torch.Tensor]


def reorder_molecular_graph(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    species: Optional[torch.Tensor] = None,
) -> ReorderedGraph:
    """Sort nodes by ``(species, morton)`` and edges lexicographically by ``(dst, src)``."""
    n = positions.shape[0]
    morton = morton_encode_3d(positions)
    perm = torch.argsort(morton, stable=True)
    if species is not None:
        sp_sorted = species[perm]
        perm = perm[torch.argsort(sp_sorted, stable=True)]

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n, device=perm.device, dtype=perm.dtype)

    new_pos = positions[perm]
    new_sp = species[perm] if species is not None else None

    src = inv_perm[edge_index[0]]
    dst = inv_perm[edge_index[1]]
    edge_order = torch.argsort(src, stable=True)
    dst_s = dst[edge_order]
    edge_order = edge_order[torch.argsort(dst_s, stable=True)]
    new_edges = torch.stack([src[edge_order], dst[edge_order]], dim=0)

    return ReorderedGraph(
        perm=perm,
        inv_perm=inv_perm,
        positions=new_pos,
        edge_index=new_edges,
        species=new_sp,
    )
