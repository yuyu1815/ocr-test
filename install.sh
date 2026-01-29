#!/bin/bash
set -e

echo "[INFO] Starting installation..."

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    pip3 install uv
else
    echo "[INFO] uv is already installed."
fi

# 2. Create virtual environment (optional but recommended)
# subprocess inside python usually uses the same env, but explicit venv is safer.
# However, for "one shot" on Jetson, sometimes --system is preferred if torch is system-wide.
# We will create a venv to keep it clean as requested by standard uv usage.
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    uv venv
fi

# Activate venv for subsequent commands in this script
source .venv/bin/activate

echo "[INFO] Installing dependencies..."

# 3. Install dependencies
# Torch is critical. on Jetson, pip install torch might pull a cpu version or fail.
# We assume the user has a way to get torch or we try standard pip.
# Checking if torch is installed:
if ! python3 -c "import torch" &> /dev/null; then
    echo "[WARNING] torch not found. Attempting standard install (might not be GPU optimized for Jetson)..."
    uv pip install torch torchvision
fi

# Install Transformers from source (needed for LightOnOCR-2)
echo "[INFO] Installing Transformers from source..."
uv pip install git+https://github.com/huggingface/transformers

# Install other requirements
echo "[INFO] Installing other libraries..."
uv pip install pillow pypdfium2 accelerate protobuf scipy opencv-python

echo "[INFO] Installation complete!"
echo ""
echo "To run the script, use:"
echo "source .venv/bin/activate"
echo "python3 receipt_ocr.py"
