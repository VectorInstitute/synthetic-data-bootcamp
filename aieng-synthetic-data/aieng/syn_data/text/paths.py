"""Path helpers for notebook and script execution."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up to a directory with pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "implementations" / "qa_text_generation"
        ).exists():
            return candidate
    return current


def use_repo_root(start: Path | None = None) -> Path:
    """Change the working directory to the repository root."""
    root = find_repo_root(start)
    os.chdir(root)
    return root
