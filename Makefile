# nvProbe Makefile — common operations

.PHONY: install dev test lint run report clean

install:
	pip install -e .

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
	python -m nvprobe.cli run --config configs/default.yaml

dry-run:
	python -m nvprobe.cli run --config configs/default.yaml --dry-run

slurm-generate:
	python -m nvprobe.cli slurm --config configs/default.yaml --action generate

slurm-submit:
	python -m nvprobe.cli slurm --config configs/default.yaml --action submit

slurm-monitor:
	python -m nvprobe.cli slurm --config configs/default.yaml --action monitor

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
