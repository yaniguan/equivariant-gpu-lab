"""Unit tests for O(3) FP8 invariance, zero-safe autograd, PBC virial, norms, and multipoles."""

from __future__ import annotations

import math

import torch

from equivariant_gpu_lab.fp8 import (
    equivariant_fp8_dequantize,
    equivariant_fp8_quantize,
    equivariant_fp8_ste,
)
from equivariant_gpu_lab.multipoles import (
    rank2_cartesian_to_irreps,
    rank2_irreps_to_cartesian,
)
from equivariant_gpu_lab.norms import EquivariantRMSNorm
from equivariant_gpu_lab.rotations import chebyshev_trig_ladder, vectors_to_zyz_angles
from equivariant_gpu_lab.safe_ops import safe_norm, safe_normalize
from equivariant_gpu_lab.virial import apply_pbc_strain, compute_conservative_virial


def _random_so3(seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


def test_zero_safe_second_order_autograd() -> None:
    vecs = torch.tensor([[0.0, 0.0, 0.0], [1.5, -0.5, 2.0]], dtype=torch.float64, requires_grad=True)
    norms = safe_norm(vecs)
    unit = safe_normalize(vecs)
    loss = (norms.sum() + (unit * unit).sum())
    (g1,) = torch.autograd.grad(loss, vecs, create_graph=True)
    assert torch.isfinite(g1).all()
    (g2,) = torch.autograd.grad(g1.sum(), vecs)
    assert torch.isfinite(g2).all()


def test_pole_safe_angles_and_chebyshev() -> None:
    poles = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    beta, alpha = vectors_to_zyz_angles(poles)
    ladder = chebyshev_trig_ladder(alpha, lmax=4)
    (grad,) = torch.autograd.grad((beta.sum() + ladder.sum()), poles)
    assert torch.isfinite(grad).all()

    theta = torch.linspace(-math.pi, math.pi, 17, dtype=torch.float64)
    trig = chebyshev_trig_ladder(theta, lmax=3)
    expected = torch.stack(
        [
            torch.cos(3 * theta),
            torch.cos(2 * theta),
            torch.cos(theta),
            torch.ones_like(theta),
            torch.sin(theta),
            torch.sin(2 * theta),
            torch.sin(3 * theta),
        ],
        dim=-1,
    )
    assert torch.allclose(trig, expected, atol=1e-12)


def test_fp8_euclidean_scale_is_so3_invariant() -> None:
    rot = _random_so3(7)
    vecs = torch.randn(32, 3, dtype=torch.float32)
    vecs_rot = vecs @ rot.T
    _, scales_a = equivariant_fp8_quantize(vecs, [3])
    _, scales_b = equivariant_fp8_quantize(vecs_rot, [3])
    assert torch.allclose(scales_a, scales_b, rtol=1e-5, atol=1e-6)

    x = torch.randn(8, 4, requires_grad=True)
    y = equivariant_fp8_ste(x, [1, 3])
    y.sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    q, sc = equivariant_fp8_quantize(x.detach(), [1, 3])
    dq = equivariant_fp8_dequantize(q, sc, [1, 3])
    assert dq.shape == x.shape


def test_pbc_virial_matches_finite_difference() -> None:
    pos = torch.tensor([[0.1, 0.2, 0.3], [1.4, -0.3, 0.8], [-0.7, 1.1, -0.5]], dtype=torch.float64, requires_grad=True)
    cell = torch.diag(torch.tensor([4.0, 4.5, 5.0], dtype=torch.float64))
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    shifts = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.long)
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)

    _, _, r_ij = apply_pbc_strain(pos, cell, edges, shifts, strain)
    dist2 = (r_ij * r_ij).sum(dim=-1)
    energy = 0.5 * ((dist2 - 2.0) ** 2).sum()
    out = compute_conservative_virial(energy, pos, strain, cell)

    eps = 1e-6
    delta = torch.zeros((3, 3), dtype=torch.float64)
    delta[0, 1] = eps
    delta[1, 0] = eps
    _, _, r_p = apply_pbc_strain(pos.detach(), cell, edges, shifts, delta)
    _, _, r_m = apply_pbc_strain(pos.detach(), cell, edges, shifts, -delta)
    e_p = 0.5 * (((r_p * r_p).sum(dim=-1) - 2.0) ** 2).sum()
    e_m = 0.5 * (((r_m * r_m).sum(dim=-1) - 2.0) ** 2).sum()
    fd_grad_01 = (e_p - e_m) / (4.0 * eps)
    vol = torch.linalg.det(cell).abs()
    assert torch.allclose(out.stress[0, 1], fd_grad_01 / vol, atol=1e-6)


def test_equivariant_rmsnorm_and_rank2_multipoles() -> None:
    rot = _random_so3(19)
    norm_mul_ir = EquivariantRMSNorm([(4, 1, True), (4, 3, False)], layout="mul_ir")
    norm_ir_mul = EquivariantRMSNorm([(4, 1, True), (4, 3, False)], layout="ir_mul")

    s = torch.randn(6, 4, 1)
    v = torch.randn(6, 4, 3)
    x_mul_ir = torch.cat([s.reshape(6, 4), v.reshape(6, 12)], dim=-1)
    x_rot = torch.cat([s.reshape(6, 4), (v @ rot.T).reshape(6, 12)], dim=-1)

    y = norm_mul_ir(x_mul_ir)
    y_rot = norm_mul_ir(x_rot)
    y_v_rot = y[:, 4:].reshape(6, 4, 3) @ rot.T
    assert torch.allclose(y_v_rot, y_rot[:, 4:].reshape(6, 4, 3), atol=1e-5)

    x_ir_mul = torch.cat([s.transpose(1, 2).reshape(6, 4), v.transpose(1, 2).reshape(6, 12)], dim=-1)
    y_im = norm_ir_mul(x_ir_mul)
    y_im_as_mul_ir = torch.cat(
        [y_im[:, :4], y_im[:, 4:].reshape(6, 3, 4).transpose(1, 2).reshape(6, 12)],
        dim=-1,
    )
    assert torch.allclose(y, y_im_as_mul_ir, atol=1e-6)

    mat = torch.randn(10, 3, 3, dtype=torch.float64)
    irr = rank2_cartesian_to_irreps(mat)
    rec = rank2_irreps_to_cartesian(irr)
    assert torch.allclose(mat, rec, atol=1e-12)
    assert torch.allclose((mat * mat).sum(dim=(-2, -1)), (irr * irr).sum(dim=-1), atol=1e-12)
