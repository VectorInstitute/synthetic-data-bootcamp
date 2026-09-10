#!/usr/bin/env python3
"""CLI wrapper for Mapillary toy extract (same logic as Notebook 0).

Examples::

  uv run python scripts/extract_mapillary_toy.py
  uv run python scripts/extract_mapillary_toy.py --clean
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgecase_synthesis.mapillary_extract import main  # noqa: E402

if __name__ == "__main__":
    main()
