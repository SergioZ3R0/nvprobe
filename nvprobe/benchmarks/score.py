"""AI Accelerator Score — normalized performance calculator.

Implements the performance scoring methodology for AI accelerators:
  - Per-Precision Compute Throughput (C[p])
  - Memory Bandwidth (BW)
  - Performance Score (P[p]) = 1 / ((1 / I_stream[p] × (BW / f)) + (1 / C[p]))
  - Global Performance (GP) = Σp X[p] × P[p]
  - Normalized Index Score = GP / H100_baseline (501.08 = 1.00)

Usage:
    nvprobe score --bandwidth 3350 --compute fp32=51,fp16=102,fp8=204
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────

# Streaming arithmetic intensity (FLOPS per byte)
I_STREAM: dict[str, float] = {
    "fp32": 0.125,
    "bf16": 0.25,
    "fp16": 0.25,
    "fp8": 0.5,
    "int8": 0.5,
    "fp4": 1.0,
    "int4": 1.0,
}

# Weight profile for global performance calculation
WEIGHTS: dict[str, float] = {
    "fp32": 0.05,
    "bf16": 0.175,
    "fp16": 0.175,
    "int8": 0.1875,
    "fp8": 0.1875,
    "int4": 0.1125,
    "fp4": 0.1125,
}

# Memory efficiency factor (empirical)
MEMORY_EFFICIENCY_FACTOR = 0.001

# NVIDIA H100 baseline GP score
H100_BASELINE_GP = 501.08


# ── Data Classes ───────────────────────────────────────────────────────────


@dataclass
class PrecisionScore:
    """Score for a single precision level."""

    precision: str
    compute_tflops: float
    i_stream: float
    weight: float
    performance_score: float = 0.0
    weighted_score: float = 0.0


@dataclass
class AcceleratorScore:
    """Complete score breakdown for an AI accelerator."""

    name: str
    bandwidth_gbs: float
    precisions: dict[str, PrecisionScore] = field(default_factory=dict)
    global_performance: float = 0.0
    normalized_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "bandwidth_gbs": self.bandwidth_gbs,
            "global_performance": round(self.global_performance, 4),
            "normalized_score": round(self.normalized_score, 4),
            "h100_baseline": H100_BASELINE_GP,
            "precisions": {
                p.precision: {
                    "compute_tflops": p.compute_tflops,
                    "i_stream": p.i_stream,
                    "weight": p.weight,
                    "performance_score": round(p.performance_score, 4),
                    "weighted_score": round(p.weighted_score, 4),
                }
                for p in self.precisions.values()
            },
        }


# ── Calculator Functions ───────────────────────────────────────────────────


def calculate_performance_score(
    compute_tflops: float,
    bandwidth_gbs: float,
    i_stream: float,
) -> float:
    """Calculate performance score for a single precision.

    Uses Roofline-inspired model: P[p] = min(C[p], BW × I_stream[p] / f)

    Args:
        compute_tflops: Sustained compute throughput in TFLOPS
        bandwidth_gbs: Sustained memory bandwidth in GB/s
        i_stream: Streaming arithmetic intensity (bytes/FLOP)

    Returns:
        Performance score P[p] in TFLOPS
    """
    if compute_tflops <= 0 or bandwidth_gbs <= 0:
        return 0.0

    # Memory-bound ceiling: BW × I_stream / f
    memory_ceiling = bandwidth_gbs * i_stream / MEMORY_EFFICIENCY_FACTOR

    # Performance is min of compute and memory ceiling
    return min(compute_tflops, memory_ceiling)


def calculate_global_performance(
    scores: dict[str, PrecisionScore],
) -> float:
    """Calculate global performance score.

    GP = Σp X[p] × P[p]

    Args:
        scores: Dictionary of precision scores

    Returns:
        Global performance score
    """
    gp = 0.0
    for precision, score in scores.items():
        gp += score.weight * score.performance_score
    return gp


def calculate_normalized_score(global_performance: float) -> float:
    """Calculate normalized score relative to H100 baseline.

    Index = GP / H100_baseline_GP

    Args:
        global_performance: Global performance score

    Returns:
        Normalized score (H100 = 1.00)
    """
    return global_performance / H100_BASELINE_GP


def score_accelerator(
    name: str,
    bandwidth_gbs: float,
    compute_by_precision: dict[str, float],
) -> AcceleratorScore:
    """Calculate complete score for an AI accelerator.

    Args:
        name: Accelerator name (e.g., "H100", "B200")
        bandwidth_gbs: Sustained memory bandwidth in GB/s
        compute_by_precision: Dictionary mapping precision to TFLOPS

    Returns:
        AcceleratorScore with all calculations
    """
    result = AcceleratorScore(name=name, bandwidth_gbs=bandwidth_gbs)

    for precision, tflops in compute_by_precision.items():
        i_stream = I_STREAM.get(precision, 0.0)
        weight = WEIGHTS.get(precision, 0.0)

        perf_score = calculate_performance_score(tflops, bandwidth_gbs, i_stream)
        weighted = weight * perf_score

        result.precisions[precision] = PrecisionScore(
            precision=precision,
            compute_tflops=tflops,
            i_stream=i_stream,
            weight=weight,
            performance_score=perf_score,
            weighted_score=weighted,
        )

    result.global_performance = calculate_global_performance(result.precisions)
    result.normalized_score = calculate_normalized_score(result.global_performance)

    return result


def compare_accelerators(
    accelerators: list[AcceleratorScore],
) -> dict[str, Any]:
    """Compare multiple accelerators and rank them.

    Args:
        accelerators: List of AcceleratorScore objects

    Returns:
        Comparison results with rankings
    """
    ranked = sorted(accelerators, key=lambda a: a.normalized_score, reverse=True)

    return {
        "rankings": [
            {
                "rank": i + 1,
                "name": a.name,
                "normalized_score": round(a.normalized_score, 4),
                "global_performance": round(a.global_performance, 4),
                "bandwidth_gbs": a.bandwidth_gbs,
            }
            for i, a in enumerate(ranked)
        ],
        "baseline": "NVIDIA H100",
        "baseline_gp": H100_BASELINE_GP,
    }
