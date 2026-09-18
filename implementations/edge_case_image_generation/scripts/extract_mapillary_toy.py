#!/usr/bin/env python3
"""CLI wrapper for Mapillary toy extract (same logic as Notebook 0).

Examples::

  uv run python scripts/extract_mapillary_toy.py
  uv run python scripts/extract_mapillary_toy.py --clean
"""

from __future__ import annotations

from aieng.syn_data.image.data.mapillary_extract import main


if __name__ == "__main__":
    main()
