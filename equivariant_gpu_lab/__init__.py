"""Equivariant GPU Lab: O(3)-equivariant primitives and GPU roofline utilities."""

from .fp8 import (
    equivariant_fp8_dequantize,
    equivariant_fp8_quantize,
    equivariant_fp8_ste,
)
from .hardware import (
    GPU_SPECS,
    SM80_A100_PCIE,
    SM80_A100_SXM4,
    SM89_L40S,
    SM90_H100_PCIE,
    SM90_H100_SXM5,
    GPUSpec,
)
from .locality import ReorderedGraph, reorder_molecular_graph
from .morton import morton_encode_3d
from .multipoles import rank2_cartesian_to_irreps, rank2_irreps_to_cartesian
from .norms import EquivariantRMSNorm
from .roofline import RooflineReport, analyze_kernel_roofline, compare_a100_to_h100_speedup
from .rotations import chebyshev_trig_ladder, vectors_to_zyz_angles
from .safe_ops import safe_norm, safe_normalize
from .segments import UniformSplitPlan, compute_uniform_split
from .virial import VirialOutput, apply_pbc_strain, compute_conservative_virial

__version__ = "0.1.0"

__all__ = [
    "EquivariantRMSNorm",
    "GPUSpec",
    "GPU_SPECS",
    "ReorderedGraph",
    "RooflineReport",
    "SM80_A100_PCIE",
    "SM80_A100_SXM4",
    "SM89_L40S",
    "SM90_H100_PCIE",
    "SM90_H100_SXM5",
    "UniformSplitPlan",
    "VirialOutput",
    "analyze_kernel_roofline",
    "apply_pbc_strain",
    "chebyshev_trig_ladder",
    "compare_a100_to_h100_speedup",
    "compute_conservative_virial",
    "compute_uniform_split",
    "equivariant_fp8_dequantize",
    "equivariant_fp8_quantize",
    "equivariant_fp8_ste",
    "morton_encode_3d",
    "rank2_cartesian_to_irreps",
    "rank2_irreps_to_cartesian",
    "reorder_molecular_graph",
    "safe_norm",
    "safe_normalize",
    "vectors_to_zyz_angles",
]
