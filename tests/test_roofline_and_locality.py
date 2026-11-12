"""Unit tests for GPU Roofline classification, 3D Morton locality, and GCD segment splits."""

from __future__ import annotations

import torch

from equivariant_gpu_lab.hardware import SM80_A100_SXM4, SM90_H100_SXM5
from equivariant_gpu_lab.locality import reorder_molecular_graph
from equivariant_gpu_lab.morton import morton_encode_3d
from equivariant_gpu_lab.roofline import (
    analyze_kernel_roofline,
    compare_a100_to_h100_speedup,
)
from equivariant_gpu_lab.segments import compute_uniform_split


def test_roofline_a100_vs_h100_regimes() -> None:
    # Memory-bound gather-scatter tensor product (AI ~ 8 FLOP/B)
    mem_rep = analyze_kernel_roofline("channelwise_tp", flops=8_000_000, bytes_transferred=1_000_000, gpu=SM90_H100_SXM5)
    assert mem_rep.regime == "memory_bound"
    mem_speedup = compare_a100_to_h100_speedup(8_000_000, 1_000_000, "bf16", "bf16")
    assert abs(mem_speedup - (SM90_H100_SXM5.hbm_gbps / SM80_A100_SXM4.hbm_gbps)) < 1e-3

    # Compute-bound triangle multiplicative update (AI ~ 600 FLOP/B)
    cmp_rep = analyze_kernel_roofline("triangle_mul", flops=600_000_000, bytes_transferred=1_000_000, gpu=SM90_H100_SXM5)
    assert cmp_rep.regime == "compute_bound"
    fp8_speedup = compare_a100_to_h100_speedup(600_000_000, 1_000_000, "bf16", "fp8")
    assert fp8_speedup > 6.0


def test_morton_and_graph_locality_invariance() -> None:
    gen = torch.Generator().manual_seed(123)
    pos = torch.rand(128, 3, generator=gen, dtype=torch.float64) * 15.0
    species = torch.randint(0, 4, (128,), generator=gen)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = torch.linalg.vector_norm(diff, dim=-1)
    mask = (dist > 0.0) & (dist < 4.0)
    edge_index = mask.nonzero(as_tuple=False).T.contiguous()

    keys = morton_encode_3d(pos)
    assert keys.shape == (128,) and keys.dtype == torch.int64

    reordered = reorder_molecular_graph(pos, edge_index, species=species)
    # Species are non-decreasing (ready for IndexedLinear CSR grouping)
    assert reordered.species is not None
    assert (reordered.species[1:] >= reordered.species[:-1]).all()
    # Destination indices are non-decreasing within reordered edges
    assert (reordered.edge_index[1, 1:] >= reordered.edge_index[1, :-1]).all()

    # Pairwise distance sum is strictly permutation-invariant
    d_orig = torch.linalg.vector_norm(pos[edge_index[1]] - pos[edge_index[0]], dim=-1).sum()
    d_reord = torch.linalg.vector_norm(
        reordered.positions[reordered.edge_index[1]] - reordered.positions[reordered.edge_index[0]],
        dim=-1,
    ).sum()
    assert torch.allclose(d_orig, d_reord, atol=1e-12)


def test_segment_uniform_split_cost_guard() -> None:
    # Tapered MACE/NequIP irreps: 128x0e + 64x1o + 32x2e -> gcd = 32
    plan = compute_uniform_split([128, 64, 32], min_extent=16, max_expansion_factor=8.0)
    assert plan.uniform is True
    assert plan.chunk_extent == 32
    assert plan.sub_counts == [4, 2, 1]

    # Prime extents -> gcd = 1 < min_extent -> rejected by cost guard
    bad = compute_uniform_split([128, 3], min_extent=16, max_expansion_factor=8.0)
    assert bad.uniform is False
