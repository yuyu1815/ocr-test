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

# 2. Sync dependencies using uv
# This command automatically creates .venv and installs everything in pyproject.toml
echo "[INFO] Syncing environment with uv..."
uv sync --prerelease=allow

echo "[INFO] Installation complete!"
echo ""
echo "To run the script, use:"
echo "source .venv/bin/activate"
echo "python3 receipt_ocr.py"
