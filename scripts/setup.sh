#!/bin/bash
set -euo pipefail

cd aieng-synthetic-data

# Ollama is installed in the Docker image but started manually when needed:
#   bash scripts/start-ollama.sh
# (from the repository root). Not all reference implementations need it.

if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    uv venv .venv
fi

source .venv/bin/activate
echo "Syncing dependencies including the text-sft group (notebook 05 LoRA)..."
uv sync --dev --group text-sft

echo "Virtual environment activated and dependencies synced."

# Install Jupyter kernel
uv run ipython kernel install --user --name=aieng-synthetic-data --display-name "AIEng Synthetic Data Bootcamp"
echo "Jupyter kernel installed."

# Start Jupyter lab
echo "Starting Jupyter lab..."
uv run jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token=''
