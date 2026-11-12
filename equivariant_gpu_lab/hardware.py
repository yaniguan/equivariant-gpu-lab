"""NVIDIA Ampere / Ada / Hopper GPU hardware specifications for Roofline modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    """Peak memory bandwidth and compute throughput (TFLOP/s) for a GPU SKU."""

    name: str
    arch: str
    hbm_gbps: float
    fp64_tflops: float
    fp32_tflops: float
    tf32_tflops: float
    bf16_tflops: float
    fp8_tflops: float = 0.0

    def peak_tflops(self, precision: str) -> float:
        table = {
            "fp64": self.fp64_tflops,
            "fp32": self.fp32_tflops,
            "tf32": self.tf32_tflops,
            "bf16": self.bf16_tflops,
            "fp16": self.bf16_tflops,
            "fp8": self.fp8_tflops,
        }
        key = precision.lower()
        if key not in table or table[key] <= 0.0:
            raise ValueError(f"Unsupported precision '{precision}' on {self.name}")
        return table[key]

    def ridge_point(self, precision: str) -> float:
        """Arithmetic intensity (FLOP / Byte) where HBM meets peak compute."""
        return (self.peak_tflops(precision) * 1e12) / (self.hbm_gbps * 1e9)


SM80_A100_SXM4 = GPUSpec(
    name="A100-SXM4-80GB",
    arch="sm_80",
    hbm_gbps=2039.0,
    fp64_tflops=19.5,
    fp32_tflops=19.5,
    tf32_tflops=156.0,
    bf16_tflops=312.0,
    fp8_tflops=0.0,
)

SM80_A100_PCIE = GPUSpec(
    name="A100-PCIe-80GB",
    arch="sm_80",
    hbm_gbps=1935.0,
    fp64_tflops=19.5,
    fp32_tflops=19.5,
    tf32_tflops=156.0,
    bf16_tflops=312.0,
    fp8_tflops=0.0,
)

SM89_L40S = GPUSpec(
    name="L40S-48GB",
    arch="sm_89",
    hbm_gbps=864.0,
    fp64_tflops=1.4,
    fp32_tflops=91.6,
    tf32_tflops=183.0,
    bf16_tflops=362.0,
    fp8_tflops=733.0,
)

SM90_H100_SXM5 = GPUSpec(
    name="H100-SXM5-80GB",
    arch="sm_90",
    hbm_gbps=3350.0,
    fp64_tflops=67.0,
    fp32_tflops=67.0,
    tf32_tflops=494.5,
    bf16_tflops=989.0,
    fp8_tflops=1979.0,
)

SM90_H100_PCIE = GPUSpec(
    name="H100-PCIe-80GB",
    arch="sm_90",
    hbm_gbps=2000.0,
    fp64_tflops=51.0,
    fp32_tflops=51.0,
    tf32_tflops=378.0,
    bf16_tflops=756.0,
    fp8_tflops=1513.0,
)

GPU_SPECS = {
    "A100_SXM4": SM80_A100_SXM4,
    "A100_PCIE": SM80_A100_PCIE,
    "L40S": SM89_L40S,
    "H100_SXM5": SM90_H100_SXM5,
    "H100_PCIE": SM90_H100_PCIE,
}
