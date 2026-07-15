"""Benchmark modules for nvProbe."""

from nvprobe.benchmarks.bandwidth import BandwidthBenchmark
from nvprobe.benchmarks.custom import CustomCudaBenchmark
from nvprobe.benchmarks.hpl import HplBenchmark
from nvprobe.benchmarks.hpcg import HpcgBenchmark
from nvprobe.benchmarks.mlperf import MlperfBenchmark

BENCHMARK_REGISTRY: dict[str, type] = {
    "bandwidth": BandwidthBenchmark,
    "custom": CustomCudaBenchmark,
    "hpl": HplBenchmark,
    "hpcg": HpcgBenchmark,
    "mlperf": MlperfBenchmark,
}

__all__ = [
    "BENCHMARK_REGISTRY",
    "BandwidthBenchmark",
    "CustomCudaBenchmark",
    "HplBenchmark",
    "HpcgBenchmark",
    "MlperfBenchmark",
]
