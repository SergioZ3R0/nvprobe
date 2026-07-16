# nvProbe Makefile — common operations

.PHONY: install dev test lint run report clean install-cupy setup

install:
	pip install -e .

setup: install install-cupy
	@echo "nvProbe ready."

install-cupy:
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1); \
	if [ -z "$$CUDA_VERSION" ]; then \
		CUDA_VERSION=$$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1); \
	fi; \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "Error: Could not detect CUDA version. Is the NVIDIA driver installed?"; \
		exit 1; \
	fi; \
	CUDA_MAJOR=$$(echo "$$CUDA_VERSION" | cut -d. -f1); \
	echo "Detected CUDA $$CUDA_VERSION (major: $$CUDA_MAJOR)"; \
	if [ "$$CUDA_MAJOR" -ge 13 ]; then \
		echo "Installing cupy-cuda13x[ctk] (includes cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE)..."; \
		pip install "cupy-cuda13x[ctk]>=13.0"; \
	elif [ "$$CUDA_MAJOR" -ge 12 ]; then \
		echo "Installing cupy-cuda12x[ctk] (includes cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE)..."; \
		pip install "cupy-cuda12x[ctk]>=13.0"; \
	elif [ "$$CUDA_MAJOR" -ge 11 ]; then \
		echo "Installing cupy-cuda11x[ctk] (includes cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE)..."; \
		pip install "cupy-cuda11x[ctk]>=12.0"; \
	else \
		echo "Error: CUDA $$CUDA_VERSION is too old. Requires CUDA 11.0+"; \
		exit 1; \
	fi

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check nvprobe/
	ruff format nvprobe/ --check

format:
	ruff format nvprobe/

run:
	python -m nvprobe.cli run --config nvprobe/configs/default.yaml

dry-run:
	python -m nvprobe.cli run --config nvprobe/configs/default.yaml --dry-run

slurm-generate:
	python -m nvprobe.cli slurm --config nvprobe/configs/default.yaml --action generate

slurm-submit:
	python -m nvprobe.cli slurm --config nvprobe/configs/default.yaml --action submit

slurm-monitor:
	python -m nvprobe.cli slurm --config nvprobe/configs/default.yaml --action monitor

report:
	python -m nvprobe.cli report

env:
	python -m nvprobe.cli env

clean:
	rm -rf results/ reports/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

container-build:
	singularity build nvprobe.sif containers/Singularity.def

container-run:
	singularity run --nv nvprobe.sif run --config configs/default.yaml
