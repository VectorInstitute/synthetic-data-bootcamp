#!/usr/bin/env bash
# One-time setup: install deps + register the notebook kernel.
#
# Uses Homebrew Python 3.12 for the venv so Cursor/VS Code can resolve the
# interpreter (uv-managed Python symlinks often fail in the IDE picker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BREW_PYTHON="/opt/homebrew/bin/python3.12"
if [[ ! -x "$BREW_PYTHON" ]]; then
  echo "Homebrew Python 3.12 not found. Install it with:"
  echo "  brew install python@3.12"
  exit 1
fi

echo "==> Creating venv with Homebrew Python..."
uv venv --python "$BREW_PYTHON" --clear

echo "==> Installing dependencies (uv sync)..."
uv sync

VENV_PYTHON="$ROOT/.venv/bin/python"

echo "==> Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install --user \
  --name edgecase-synthesis \
  --display-name "EdgeCase Synthesis (.venv)"

echo ""
echo "Done. Select this interpreter in Cursor:"
echo "  $VENV_PYTHON"
echo ""
echo "If the picker still fails, use the direct Homebrew path:"
echo "  $BREW_PYTHON"
echo ""
echo "Then in the notebook: Select Kernel → EdgeCase Synthesis (.venv)"
