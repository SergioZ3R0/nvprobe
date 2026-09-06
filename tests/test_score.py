"""Tests for AI Accelerator Score calculator."""

from nvprobe.benchmarks.score import (
    H100_BASELINE_GP,
    AcceleratorScore,
    calculate_global_performance,
    calculate_normalized_score,
    calculate_performance_score,
    compare_accelerators,
    score_accelerator,
)


def test_performance_score_basic() -> None:
    """Performance score should be positive for valid inputs."""
    score = calculate_performance_score(
        compute_tflops=51.0,
        bandwidth_gbs=3350.0,
        i_stream=0.125,
    )
    assert score > 0


def test_performance_score_zero_compute() -> None:
    """Performance score should be 0 when compute is 0."""
    score = calculate_performance_score(
        compute_tflops=0.0,
        bandwidth_gbs=3350.0,
        i_stream=0.125,
    )
    assert score == 0.0


def test_performance_score_zero_bandwidth() -> None:
    """Performance score should be 0 when bandwidth is 0."""
    score = calculate_performance_score(
        compute_tflops=51.0,
        bandwidth_gbs=0.0,
        i_stream=0.125,
    )
    assert score == 0.0


def test_performance_score_higher_compute_higher_score() -> None:
    """Higher compute should yield higher performance score."""
    low = calculate_performance_score(50.0, 3350.0, 0.125)
    high = calculate_performance_score(100.0, 3350.0, 0.125)
    assert high > low


def test_performance_score_higher_bandwidth_higher_score() -> None:
    """Higher bandwidth should yield higher score when memory-bound."""
    # Roofline model: P = min(C, BW * I / f)
    # When compute-bound (C < BW*I/f), score = C regardless of BW
    # When memory-bound (C > BW*I/f), score = BW*I/f
    # Test memory-bound case: very high compute, low BW
    low = calculate_performance_score(1000.0, 100.0, 0.125)
    high = calculate_performance_score(1000.0, 200.0, 0.125)
    # memory_ceiling_low = 100 * 0.125 / 0.001 = 12500
    # memory_ceiling_high = 200 * 0.125 / 0.001 = 25000
    # Both > 1000, so both return 1000 (compute-bound)
    # Test with very high compute to ensure memory-bound
    low = calculate_performance_score(100000.0, 100.0, 0.125)
    high = calculate_performance_score(100000.0, 200.0, 0.125)
    # memory_ceiling_low = 100 * 0.125 / 0.001 = 12500
    # memory_ceiling_high = 200 * 0.125 / 0.001 = 25000
    # Both < 100000, so memory-bound
    assert low == 12500.0
    assert high == 25000.0
    assert high > low


def test_global_performance_weighted_sum() -> None:
    """Global performance should be weighted sum of precision scores."""
    from nvprobe.benchmarks.score import PrecisionScore, WEIGHTS

    scores = {}
    for prec in ["fp32", "fp16"]:
        perf = calculate_performance_score(51.0, 3350.0, 0.125)
        scores[prec] = PrecisionScore(
            precision=prec,
            compute_tflops=51.0,
            i_stream=0.125,
            weight=WEIGHTS[prec],
            performance_score=perf,
            weighted_score=WEIGHTS[prec] * perf,
        )

    gp = calculate_global_performance(scores)
    expected = sum(s.weighted_score for s in scores.values())
    assert abs(gp - expected) < 1e-10


def test_normalized_score_h100_baseline() -> None:
    """H100 baseline should normalize to 1.0."""
    normalized = calculate_normalized_score(H100_BASELINE_GP)
    assert abs(normalized - 1.0) < 1e-6


def test_normalized_score_higher_than_h100() -> None:
    """Score higher than H100 baseline should normalize to > 1.0."""
    normalized = calculate_normalized_score(H100_BASELINE_GP * 2)
    assert normalized > 1.0


def test_score_accelerator_complete() -> None:
    """score_accelerator should return complete score breakdown."""
    result = score_accelerator(
        name="H100",
        bandwidth_gbs=3350.0,
        compute_by_precision={"fp32": 51.0, "fp16": 102.0},
    )

    assert result.name == "H100"
    assert result.bandwidth_gbs == 3350.0
    assert "fp32" in result.precisions
    assert "fp16" in result.precisions
    assert result.global_performance > 0
    assert result.normalized_score > 0


def test_score_accelerator_to_dict() -> None:
    """to_dict should return serializable dictionary."""
    result = score_accelerator(
        name="B200",
        bandwidth_gbs=8000.0,
        compute_by_precision={"fp32": 90.0, "fp16": 180.0},
    )

    d = result.to_dict()
    assert d["name"] == "B200"
    assert d["bandwidth_gbs"] == 8000.0
    assert "fp32" in d["precisions"]
    assert "normalized_score" in d
    assert "global_performance" in d


def test_compare_accelerators_ranking() -> None:
    """compare_accelerators should rank by normalized score."""
    acc1 = score_accelerator("Slow", 2000.0, {"fp32": 30.0})
    acc2 = score_accelerator("Fast", 4000.0, {"fp32": 60.0})

    result = compare_accelerators([acc1, acc2])
    rankings = result["rankings"]

    assert len(rankings) == 2
    assert rankings[0]["name"] == "Fast"
    assert rankings[1]["name"] == "Slow"
    assert rankings[0]["rank"] == 1
    assert rankings[1]["rank"] == 2


def test_compare_accelerators_empty() -> None:
    """compare_accelerators should handle empty list."""
    result = compare_accelerators([])
    assert result["rankings"] == []


def test_h100_baseline_value() -> None:
    """H100 baseline should be 501.08."""
    assert H100_BASELINE_GP == 501.08
