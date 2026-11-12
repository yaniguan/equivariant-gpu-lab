"""Periodic boundary condition (PBC) symmetric strain, conservative forces, and virial stress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class VirialOutput:
    forces: torch.Tensor
    virial: torch.Tensor
    stress: torch.Tensor
    pressure: torch.Tensor


def apply_pbc_strain(
    positions: torch.Tensor,
    cell: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shifts: torch.Tensor,
    strain: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply symmetric strain ``I + 0.5*(eps + eps^T)`` to atomic positions, cell, and PBC edges."""
    sym_eps = 0.5 * (strain + strain.transpose(-1, -2))
    deform = torch.eye(3, dtype=positions.dtype, device=positions.device) + sym_eps
    strained_pos = positions @ deform.T
    strained_cell = cell @ deform.T
    src, dst = edge_index[0], edge_index[1]
    edge_vecs = strained_pos[dst] - strained_pos[src] + edge_shifts.to(positions.dtype) @ strained_cell
    return strained_pos, strained_cell, edge_vecs


def compute_conservative_virial(
    energy: torch.Tensor,
    positions: torch.Tensor,
    strain: torch.Tensor,
    cell: torch.Tensor,
    create_graph: bool = False,
) -> VirialOutput:
    """Differentiate total potential energy wrt positions and symmetric strain."""
    grad_pos, dE_deps = torch.autograd.grad(
        energy.sum(),
        [positions, strain],
        create_graph=create_graph,
    )
    forces = -grad_pos
    sym_dE = 0.5 * (dE_deps + dE_deps.transpose(-1, -2))
    virial = -sym_dE
    volume = torch.linalg.det(cell).abs()
    stress = sym_dE / volume
    pressure = -torch.diagonal(stress, dim1=-2, dim2=-1).mean(dim=-1)
    return VirialOutput(forces=forces, virial=virial, stress=stress, pressure=pressure)
