"""Orthonormal rank-2 Cartesian tensor <-> spherical irrep (0e + 1e + 2e) bridge."""

from __future__ import annotations

import math

import torch

_INV_SQRT3 = 1.0 / math.sqrt(3.0)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT6 = 1.0 / math.sqrt(6.0)


def rank2_cartesian_to_irreps(tensor_3x3: torch.Tensor) -> torch.Tensor:
    """Orthonormal map from ``(..., 3, 3)`` Cartesian tensor to ``(..., 9)`` ``[0e(1), 1e(3), 2e(5)]``."""
    if tensor_3x3.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(tensor_3x3.shape)}")
    t = tensor_3x3
    xx, xy, xz = t[..., 0, 0], t[..., 0, 1], t[..., 0, 2]
    yx, yy, yz = t[..., 1, 0], t[..., 1, 1], t[..., 1, 2]
    zx, zy, zz = t[..., 2, 0], t[..., 2, 1], t[..., 2, 2]

    s0 = (xx + yy + zz) * _INV_SQRT3

    a_x = (yz - zy) * _INV_SQRT2
    a_y = (zx - xz) * _INV_SQRT2
    a_z = (xy - yx) * _INV_SQRT2

    q_xy = (xy + yx) * _INV_SQRT2
    q_yz = (yz + zy) * _INV_SQRT2
    q_z2 = (2.0 * zz - xx - yy) * _INV_SQRT6
    q_xz = (xz + zx) * _INV_SQRT2
    q_x2_y2 = (xx - yy) * _INV_SQRT2

    return torch.stack([s0, a_x, a_y, a_z, q_xy, q_yz, q_z2, q_xz, q_x2_y2], dim=-1)


def rank2_irreps_to_cartesian(irreps_9: torch.Tensor) -> torch.Tensor:
    """Inverse orthonormal map from ``(..., 9)`` ``[0e(1), 1e(3), 2e(5)]`` to ``(..., 3, 3)``."""
    if irreps_9.shape[-1] != 9:
        raise ValueError(f"Expected (..., 9), got {tuple(irreps_9.shape)}")
    s0 = irreps_9[..., 0]
    a_x, a_y, a_z = irreps_9[..., 1], irreps_9[..., 2], irreps_9[..., 3]
    q_xy, q_yz, q_z2, q_xz, q_x2_y2 = (
        irreps_9[..., 4],
        irreps_9[..., 5],
        irreps_9[..., 6],
        irreps_9[..., 7],
        irreps_9[..., 8],
    )

    tr = s0 * _INV_SQRT3
    xx = tr - q_z2 * _INV_SQRT6 + q_x2_y2 * _INV_SQRT2
    yy = tr - q_z2 * _INV_SQRT6 - q_x2_y2 * _INV_SQRT2
    zz = tr + 2.0 * q_z2 * _INV_SQRT6

    xy = (q_xy + a_z) * _INV_SQRT2
    yx = (q_xy - a_z) * _INV_SQRT2
    yz = (q_yz + a_x) * _INV_SQRT2
    zy = (q_yz - a_x) * _INV_SQRT2
    xz = (q_xz - a_y) * _INV_SQRT2
    zx = (q_xz + a_y) * _INV_SQRT2

    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([yx, yy, yz], dim=-1)
    row2 = torch.stack([zx, zy, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
