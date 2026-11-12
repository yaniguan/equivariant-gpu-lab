"""Analytical Roofline classification for equivariant GNN and Pairformer workloads."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import GPUSpec, SM80_A100_SXM4, SM90_H100_SXM5


@dataclass(frozen=True)
class RooflineReport:
    op_name: str
    gpu: str
    precision: str
    flops: int
    bytes_transferred: int
    arithmetic_intensity: float
    ridge_point: float
    attained_ceiling_tflops: float
    ideal_time_us: float
    regime: str


def analyze_kernel_roofline(
    op_name: str,
    flops: int,
    bytes_transferred: int,
    gpu: GPUSpec = SM90_H100_SXM5,
    precision: str = "bf16",
) -> RooflineReport:
    """Classify an operator into Memory-Bound vs Compute-Bound on a target GPU."""
    if flops <= 0 or bytes_transferred <= 0:
        raise ValueError("flops and bytes_transferred must be strictly positive")
    ai = flops / bytes_transferred
    ridge = gpu.ridge_point(precision)
    peak = gpu.peak_tflops(precision)
    bw_limited_tflops = (ai * gpu.hbm_gbps * 1e9) / 1e12
    attained = min(peak, bw_limited_tflops)
    regime = "memory_bound" if ai < ridge else "compute_bound"
    ideal_us = (flops / (attained * 1e12)) * 1e6
    return RooflineReport(
        op_name=op_name,
        gpu=gpu.name,
        precision=precision,
        flops=flops,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=ai,
        ridge_point=ridge,
        attained_ceiling_tflops=attained,
        ideal_time_us=ideal_us,
        regime=regime,
    )


def compare_a100_to_h100_speedup(
    flops: int,
    bytes_transferred: int,
    a100_precision: str = "bf16",
    h100_precision: str = "bf16",
) -> float:
    """Predict ideal Roofline speedup moving from A100-SXM4 to H100-SXM5."""
    r_a100 = analyze_kernel_roofline("op", flops, bytes_transferred, SM80_A100_SXM4, a100_precision)
    r_h100 = analyze_kernel_roofline("op", flops, bytes_transferred, SM90_H100_SXM5, h100_precision)
    return r_a100.ideal_time_us / r_h100.ideal_time_us
