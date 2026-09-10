"""Path helpers for notebook and script execution."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from aieng.syn_data.text.config import IMPLEMENTATION_DIR


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


def load_implementation_dotenv(*, override: bool = False) -> Path | None:
    """Load env vars from the reference-implementation ``.env`` file.

    Looks for ``implementations/qa_text_generation/.env`` first (canonical
    location). Falls back to a repo-root ``.env`` if present so existing
    checkouts keep working during the move.

    Returns the path that was loaded, or ``None`` if neither file exists.
    """
    candidates = (
        IMPLEMENTATION_DIR / ".env",
        find_repo_root() / ".env",
    )
    for env_path in candidates:
        if env_path.is_file():
            load_dotenv(env_path, override=override)
            return env_path
    return None
