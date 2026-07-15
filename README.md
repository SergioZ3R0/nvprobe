# nvProbe: NVIDIA GPU & CUDA Benchmark Suite

NVIDIA GPU benchmark suite for CUDA workload automation, reporting, and comparison.

nvProbe runs standardized benchmarks across your GPU fleet, captures environment details, stores results in SQLite, and generates self-contained HTML reports — helping HPC engineers and ML teams make data-driven hardware purchasing decisions.

## Features

- **Benchmark modules**: Bandwidth, custom CUDA kernels (matmul, conv2d, attention), HPL, HPCG, MLPerf
- **Slurm integration**: Generate and submit sbatch scripts, run across multiple nodes/GPUs
- **Environment fingerprinting**: Driver version, CUDA version, GPU model, memory, PCI bus ID — captured automatically
- **SQLite storage**: All results persisted with full query capability
- **CSV/JSON export**: Raw data for programmatic access
- **HTML reports**: Self-contained reports with matplotlib charts, sidebar navigation, and comparison views
- **YAML configs**: Define test matrices (GPU models, precisions, batch sizes) declaratively
- **Reproducible**: Same config + same hardware = same results

## Quick Start

```bash
git clone https://github.com/SergioZ3R0/nvprobe.git
cd nvprobe
pip install -e .
```

### Or install from PyPI

```bash
pip install nvprobe
```

### Detect GPU environment

```bash
nvprobe env
```

### Run benchmarks (dry run)

```bash
nvprobe run --config configs/default.yaml --dry-run
```

### Run benchmarks

```bash
nvprobe run --config configs/default.yaml
```

### Generate report

```bash
nvprobe report
```

### Compare two runs

```bash
nvprobe compare --a results/run1 --b results/run2
```

## Configuration

Edit `configs/default.yaml` to define your test matrix:

```yaml
name: my-benchmark-run
description: "Comparing L40S vs B200"

gpu:
  models: ["L40S", "B200"]

slurm:
  enabled: true
  partition: gpu
  gpus_per_node: 8

precisions:
  - fp32
  - fp16
  - int8

benchmarks:
  - name: bandwidth
    enabled: true
    params:
      sizes_mb: [1, 4, 16, 64, 256, 1024]
  - name: custom
    enabled: true
    params:
      kernels: [matmul, conv2d, attention]
```

## Project Structure

```
nvprobe/
├── nvprobe/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # YAML config loader
│   ├── runner.py           # Benchmark orchestration
│   ├── slurm.py            # Slurm job management
│   ├── reporter.py         # HTML report generator with charts
│   ├── db.py               # SQLite storage + CSV/JSON export
│   └── benchmarks/
│       ├── base.py         # Base benchmark class
│       ├── bandwidth.py    # Memory bandwidth tests
│       ├── custom.py       # Custom CUDA kernels
│       ├── hpl.py          # HPL wrapper
│       ├── hpcg.py         # HPCG wrapper
│       ├── mlperf.py       # MLPerf wrapper
│       └── _cuda/
│           ├── bandwidth_test.py   # CUDA bandwidth implementation
│           ├── custom_kernels.py   # matmul/conv2d/attention
│           └── utils.py            # Shared GPU utilities
├── configs/
│   └── default.yaml        # Default test configuration
├── reports/                 # Generated HTML reports
├── results/                 # Benchmark results (SQLite + JSON)
├── README.md
└── pyproject.toml
```

## Roadmap

### v0.1.0 — Project base ✓
- CLI with Typer (run, report, compare, env, version)
- YAML config system for test matrices
- Benchmark module framework (base class + stubs)
- Runner with nvidia-smi environment detection
- SQLite storage for results
- HTML report generator (basic)
- Default config for L40S/B200 GPUs

### v0.2.0 — CUDA benchmarks ✓
- Bandwidth test (host↔device, device↔device) via cupy
- Custom CUDA kernels: matmul, conv2d, attention
- HPL/HPCG binary wrappers with Slurm script generation
- MLPerf inference/training wrapper
- Optional cupy dependency (`pip install nvprobe[cuda]`)

### v0.3.0 — Slurm integration ✓
- sbatch script generation
- Job submission and monitoring
- Multi-GPU parallel execution
- Result collection from Slurm output

### v0.4.0 — Reporting ✓
- Matplotlib charts (bandwidth, matmul, attention, GPU comparison)
- Corporate branding (sidebar, color palette, env cards)
- Comparison reports (A vs B)
- CSV/JSON auto-export alongside HTML

### v0.5.0 — Reproducibility ✓
- Singularity container support (CUDA 12.4 runtime)
- Makefile for common dev operations
- Git-tracked configs and results
- Singularity container support
- Environment fingerprinting
- Git-tracked configs and results

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA drivers installed
- Slurm (for multi-node execution)
- `nvidia-smi` available in PATH
- Singularity (optional, for containerized execution)

## Container

```bash
make container-build    # builds nvprobe.sif
make container-run      # runs with --nv GPU passthrough
```

## License

[Apache License 2.0](LICENSE)
