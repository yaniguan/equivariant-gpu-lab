# equivariant-gpu-lab

Lightweight, dependency-minimal reference primitives and analytical Roofline utilities for $O(3)$-equivariant neural interatomic potentials (MACE, NequIP, eSCN) and geometric Pairformer workloads on NVIDIA Ampere (`sm_80` A100) and Hopper (`sm_90` H100) GPUs.

## Modules

| Module | Description |
| :--- | :--- |
| `hardware` / `roofline` | Analytical Roofline ridge points and A100-SXM4 vs H100-SXM5 speedup estimators across `FP64`, `FP32`, `TF32`, `BF16`, and `FP8` |
| `morton` / `locality` | Vectorized 63-bit 3D Morton space-filling curve keys and `(species, morton)` / `(dst, src)` graph reordering for L2 cache locality |
| `safe_ops` | Double-`where` zero-safe Euclidean norms and unit-vector normalization with finite 1st/2nd-order derivatives at $r = 0$ |
| `rotations` | Pole-safe `atan2` ZYZ Euler angle extraction and algebraic Chebyshev harmonic recurrence |
| `fp8` | Hopper $O(3)$-invariant Euclidean-$L_2$ per-irrep `float8_e4m3fn` block quantization and STE |
| `virial` | Periodic boundary condition (PBC) symmetric strain deformation and conservative atomic forces / Cauchy stress |
| `segments` | Cost-guarded GCD segment uniformizer for tapered irrep multiplicities (`128x0e + 64x1o + 32x2e`) |
| `norms` / `multipoles` | Dual-layout (`mul_ir`, `ir_mul`) `EquivariantRMSNorm` and orthonormal rank-2 Cartesian $\leftrightarrow$ `0e + 1e + 2e` spherical tensor bridge |

## Quick Verification

```bash
pip install -e ".[test]"
pytest -v
```
