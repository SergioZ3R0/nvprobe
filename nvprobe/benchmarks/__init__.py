"""Benchmark modules for nvProbe."""

from nvprobe.benchmarks.bandwidth import BandwidthBenchmark
from nvprobe.benchmarks.burn import BurnBenchmark
from nvprobe.benchmarks.custom import CustomCudaBenchmark
from nvprobe.benchmarks.hpl import HplBenchmark
from nvprobe.benchmarks.hpcg import HpcgBenchmark
from nvprobe.benchmarks.memtest import MemtestBenchmark
from nvprobe.benchmarks.mlperf import MlperfBenchmark

BENCHMARK_REGISTRY: dict[str, type] = {
    "bandwidth": BandwidthBenchmark,
    "burn": BurnBenchmark,
    "custom": CustomCudaBenchmark,
    "hpl": HplBenchmark,
    "hpcg": HpcgBenchmark,
    "memtest": MemtestBenchmark,
    "mlperf": MlperfBenchmark,
}

__all__ = [
    "BENCHMARK_REGISTRY",
    "BandwidthBenchmark",
    "BurnBenchmark",
    "CustomCudaBenchmark",
    "HplBenchmark",
    "HpcgBenchmark",
    "MemtestBenchmark",
    "MlperfBenchmark",
]
