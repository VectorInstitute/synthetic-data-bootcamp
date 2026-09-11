"""Shared notebook / script bootstrap for the implementation root."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root(start: Path | None = None) -> Path:
    """Locate ``edge_case_image_generation/``, put ``src`` on ``sys.path``, return root.

    Matches the previous inlined notebook helper: walk cwd parents, also consider
    ``implementations/edge_case_image_generation`` under each parent, and accept a
    directory that has both ``src/edgecase_synthesis`` and ``configs/``.
    """
    here = (start or Path.cwd()).resolve()
    search = [here, *here.parents]
    for base in list(search):
        nested = base / "implementations" / "edge_case_image_generation"
        if nested.is_dir():
            search.append(nested)
    for base in search:
        if (base / "src" / "edgecase_synthesis").is_dir() and (base / "configs").is_dir():
            src = str(base / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            return base
    raise FileNotFoundError(
        f"Could not find edge_case_image_generation root (need src/edgecase_synthesis + configs/) from {here}",
    )
