"""Utility functions for singe-table tabular data generation notebooks."""

from pathlib import Path


def get_root_directory() -> Path:
    """Get the root directory of the single-table tabular data generation implementation."""
    return Path(__file__).resolve().parents[0]
