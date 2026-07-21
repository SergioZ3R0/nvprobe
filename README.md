<div align="center">
  <img src="https://raw.githubusercontent.com/SergioZ3R0/nvprobe/main/nvprobe/nvprobe.svg" alt="nvProbe" width="180">
</div>

<h1 align="center">nvProbe</h1>
<p align="center">
  <b>NVIDIA GPU &amp; CUDA Benchmark Suite</b><br>
  <i>Automate CUDA workloads &bull; HPL &amp; HPCG &bull; MLPerf inference &bull; Custom kernels &bull; Interactive reports</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/nvprobe/"><img src="https://img.shields.io/badge/PyPI-3775A9?style=flat&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/nvprobe/"><img src="https://img.shields.io/pypi/pyversions/nvprobe?style=flat&label=Python" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-12%20%7C%2013-76b900?style=flat&logo=nvidia" alt="CUDA">
</p>

```bash
pip install nvprobe && nvprobe setup && nvprobe run --local
```

---

## Features

| | | |
|---|---|---|
| **Bandwidth** | **MatMul / Attention** | **Conv2D** |
| H2D / D2H / D2D across buffer sizes | fp32, fp16, int8 custom CUDA kernels | 2D convolution benchmarks |
| **HPL** (FP64 Linpack) | **HPCG** | **MLPerf Inference** |
| Datacenter GPUs: A100, H100, B200, L40S… | Conjugate Gradients | ONNX Runtime via cmx4mlperf |

- **Bundled CUDA runtime** &mdash; CuPy `[ctk]` via pip, no system toolkit required
- **Auto-downloaded HPC tools** &mdash; NVIDIA HPC Benchmarks cached in `~/.nvprobe/tools/`
- **Interactive HTML reports** &mdash; Plotly charts with GPU / transfer / precision dropdowns
- **A/B comparison** &mdash; compare two result sets side-by-side
- **Slurm integration** &mdash; generate, submit, monitor, collect from HPC clusters
- **SQLite storage** &mdash; all results persisted; CSV / JSON export

## Quick Start

| Step | Command | What it does |
|------|---------|-------------|
| 1 | `pip install nvprobe` | Install the package |
| 2 | `nvprobe setup` | Install CuPy, download HPL/HPCG, generate configs |
| 3 | `nvprobe env` | Verify GPU detection, driver, CUDA version |
| 4 | `nvprobe run --local` | Run all benchmarks locally |
| 5 | `nvprobe report --open` | Generate &amp; open interactive HTML report |

Or from source:
```bash
git clone https://github.com/SergioZ3R0/nvprobe.git && cd nvprobe
pip install -e . && nvprobe setup && nvprobe run --local
```

### More commands

| Command | Description |
|---------|-------------|
| `nvprobe compare --a results/run1 --b results/run2` | Compare two runs |
| `nvprobe run --config configs/cluster.yaml` | Run with custom YAML config |
| `nvprobe slurm submit --config configs/cluster.yaml` | Submit Slurm job |
| `nvprobe slurm status` | Check Slurm job status |
| `nvprobe setup --cuda 13` | Setup with specific CUDA version |

## Charts

Plotly charts with interactive controls:

- **Bandwidth** &mdash; filter by GPU and transfer type (H2D / D2H / D2D)
- **MatMul / Attention** &mdash; filter by GPU and precision (fp32 / fp16)
- **Range slider** &mdash; zoom into any x-axis region
- **Moving average** &mdash; smoother trend lines for dense data

## YAML Config

```yaml
name: my-run
gpu:
  models: ["L40S", "B200"]
slurm:
  enabled: true
  partition: gpu
  gpus_per_node: 8
precisions: [fp32, fp16]
benchmarks:
  - name: bandwidth
    params:
      sizes_mb: [1, 4, 16, 64, 256, 1024]
  - name: custom
    params:
      kernels: [matmul, attention]
```

## Project Structure

```
nvprobe/
├── nvprobe/
│   ├── cli.py                     # CLI entry point
│   ├── config.py                  # YAML config loader
│   ├── runner.py                  # Benchmark orchestration
│   ├── slurm.py                   # Slurm job management
│   ├── reporter.py                # Plotly HTML report generator
│   ├── db.py                      # SQLite storage + CSV/JSON export
│   └── benchmarks/
│       ├── base.py                # Base class, GPU detection, diagnostics
│       ├── bandwidth.py           # Memory bandwidth tests
│       ├── custom.py              # Custom CUDA kernels
│       ├── hpl.py                 # HPL wrapper
│       ├── hpcg.py                # HPCG wrapper
│       ├── mlperf.py              # MLPerf via cmx4mlperf
│       └── _cuda/                 # Raw CUDA kernels
├── configs/
│   ├── default.yaml
│   └── local.yaml
├── nvprobe.svg
├── index.html
├── README.md
└── pyproject.toml
```

## Notes

- **HPL / HPCG** &mdash; NVIDIA HPC Benchmarks binaries are validated for datacenter GPUs (A100, H100, B200, L40S…). They may crash (SIGSEGV) on RTX series. Bandwidth and custom kernels work on any CUDA GPU.
- **MLPerf cuDNN** &mdash; `mlcr` discovers cuDNN via system CUDA paths. If installed via `pip install nvidia-cudnn-cuXX`, pre-register with: `mlcr get,cudnn,nvidia --input=$(python3 -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0]'))`

## Requirements

Python 3.10+ &bull; NVIDIA GPU with CUDA drivers &bull; `nvidia-smi` in PATH &bull; Slurm (optional)

## License

[Apache License 2.0](LICENSE)
