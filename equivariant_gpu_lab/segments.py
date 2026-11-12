"""Cost-guarded GCD segment uniformization for tapered irrep multiplicities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class UniformSplitPlan:
    uniform: bool
    chunk_extent: int
    original_segments: int
    expanded_segments: int
    expansion_factor: float
    sub_counts: List[int]


def compute_uniform_split(
    extents: Sequence[int],
    min_extent: int = 16,
    max_expansion_factor: float = 8.0,
) -> UniformSplitPlan:
    """Determine whether heterogeneous segment extents can be profitably split into uniform GCD chunks."""
    if not extents:
        return UniformSplitPlan(True, 0, 0, 0, 1.0, [])
    vals = [int(e) for e in extents]
    if any(v <= 0 for v in vals):
        raise ValueError("All segment extents must be positive")

    if len(set(vals)) == 1:
        return UniformSplitPlan(True, vals[0], len(vals), len(vals), 1.0, [1] * len(vals))

    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)

    counts = [v // g for v in vals]
    expanded = sum(counts)
    factor = expanded / len(vals)
    if g < min_extent or factor > max_expansion_factor:
        return UniformSplitPlan(False, g, len(vals), len(vals), 1.0, [1] * len(vals))

    return UniformSplitPlan(True, g, len(vals), expanded, factor, counts)
