"""Utility functions for agent benchmark generation notebooks."""

from pathlib import Path


def get_root_directory() -> Path:
    """Get the root directory of the agent benchmark generation implementation."""
    return Path(__file__).resolve().parents[1]
