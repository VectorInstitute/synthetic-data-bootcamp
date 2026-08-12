from pathlib import Path


def get_root_directory():
    """Get the root directory of the agent benchmark generation implementation."""
    return Path(__file__).resolve().parents[1]