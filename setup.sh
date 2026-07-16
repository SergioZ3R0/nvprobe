#!/usr/bin/env bash
# nvProbe one-liner setup: detects CUDA, installs self-contained CuPy + nvprobe
# Usage: curl -sSL <raw-url>/setup.sh | bash
#    or: bash setup.sh
set -euo pipefail

echo "=== nvProbe Setup ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Requires Python 3.10+"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Error: Python $PY_VER found, requires 3.10+"
    exit 1
fi
echo "Python: $PY_VER"

# Detect CUDA
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
fi
if [ -z "$CUDA_VERSION" ] && command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
fi
if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA. Is the NVIDIA driver installed?"
    echo "  nvidia-smi should show CUDA version"
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
echo "CUDA: $CUDA_VERSION (major: $CUDA_MAJOR)"

if [ "$CUDA_MAJOR" -lt 11 ]; then
    echo "Error: CUDA $CUDA_VERSION too old. Requires 11.0+"
    exit 1
fi

# Install nvprobe + self-contained CuPy
echo ""
echo "Installing nvprobe + cupy-cuda${CUDA_MAJOR}x[ctk]..."
pip3 install --user -e ".[cupy-cuda${CUDA_MAJOR}x]"

echo ""
echo "=== Setup complete ==="
echo "Run: nvprobe run --config configs/local.yaml --local"
