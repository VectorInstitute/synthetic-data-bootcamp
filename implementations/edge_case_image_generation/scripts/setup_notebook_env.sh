#!/usr/bin/env bash
# One-time setup: install deps from the monorepo root + register the notebook kernel.
#
# Uses Homebrew Python 3.12 for the venv so Cursor/VS Code can resolve the
# interpreter (uv-managed Python symlinks often fail in the IDE picker).
set -euo pipefail

IMPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$IMPL_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

BREW_PYTHON="/opt/homebrew/bin/python3.12"
if [[ ! -x "$BREW_PYTHON" ]]; then
  echo "Homebrew Python 3.12 not found. Install it with:"
  echo "  brew install python@3.12"
  exit 1
fi

echo "==> Creating venv at repo root with Homebrew Python..."
uv venv --python "$BREW_PYTHON" --clear

echo "==> Installing dependencies (uv sync --group edge-case-image-generation)..."
uv sync --dev --group edge-case-image-generation

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

echo "==> Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install --user \
  --name aieng-synthetic-data \
  --display-name "AIEng Synthetic Data Bootcamp"

echo ""
echo "Done. Select this interpreter in Cursor:"
echo "  $VENV_PYTHON"
echo ""
echo "If the picker still fails, use the direct Homebrew path:"
echo "  $BREW_PYTHON"
echo ""
echo "Then in the notebook: Select Kernel → AIEng Synthetic Data Bootcamp"
