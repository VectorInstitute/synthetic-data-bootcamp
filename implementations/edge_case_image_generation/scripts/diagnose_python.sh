#!/usr/bin/env bash
# Print every Python interpreter Cursor should be able to find.
set -euo pipefail

echo "=== Python on this Mac (CLI) ==="
echo "Note: macOS often has NO 'python' command — only python3, python3.12, etc."
echo ""

for cmd in python python3 python3.12 python3.14; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd -> $(command -v "$cmd") ($($cmd --version 2>&1))"
  else
    echo "$cmd -> (not on PATH)"
  fi
done

echo ""
echo "=== Project venv ==="
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python3.12"
if [[ -x "$VENV_PY" ]]; then
  echo "OK  $VENV_PY ($("$VENV_PY" --version 2>&1))"
else
  echo "MISSING  $VENV_PY — run ./scripts/setup_notebook_env.sh"
fi

echo ""
echo "=== Jupyter kernels ==="
if command -v jupyter >/dev/null 2>&1; then
  jupyter kernelspec list
else
  "$VENV_PY" -m jupyter kernelspec list 2>/dev/null || echo "jupyter not installed"
fi

echo ""
echo "=== Paste into Cursor (Python: Select Interpreter → Enter path) ==="
echo "$VENV_PY"
