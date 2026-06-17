#!/bin/bash
set -euo pipefail

cd aieng-synthetic-data

# Start Ollama and pull the Qwen 3B weights for baseline / eval notebooks.
bash scripts/start-ollama.sh

if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    uv venv .venv
fi

source .venv/bin/activate
if [ "${RUN_SFT:-0}" = "1" ]; then
    echo "RUN_SFT=1: syncing SFT dependencies (CUDA / bitsandbytes)..."
    uv sync --dev --group text-sft
else
    uv sync --dev
fi

echo "Virtual environment activated and dependencies synced."

# Install Jupyter kernel
uv run ipython kernel install --user --name=aieng-synthetic-data --display-name "AIEng Synthetic Data Bootcamp"
echo "Jupyter kernel installed."

# Start Jupyter lab
echo "Starting Jupyter lab..."
uv run jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token=''
