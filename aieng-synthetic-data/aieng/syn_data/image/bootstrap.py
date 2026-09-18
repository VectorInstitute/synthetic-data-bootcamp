"""Shared notebook / script bootstrap for the implementation root."""

from __future__ import annotations

from pathlib import Path


def _is_impl_root(candidate: Path) -> bool:
    return (
        candidate.name == "edge_case_image_generation"
        and (candidate / "configs" / "config.yaml").is_file()
    )


def bootstrap_project_root(start: Path | None = None) -> Path:
    """Locate ``implementations/edge_case_image_generation`` (Hydra configs + data).

    Package code lives in ``aieng.syn_data.image`` (via ``aieng-synthetic-data``);
    this only finds the implementation folder for configs / data / outputs.
    """
    here = (start or Path.cwd()).resolve()
    search = [here, *here.parents]
    for base in list(search):
        nested = base / "implementations" / "edge_case_image_generation"
        if nested.is_dir():
            search.append(nested)
    for base in search:
        if _is_impl_root(base):
            return base
    raise FileNotFoundError(
        f"Could not find edge_case_image_generation root "
        f"(need configs/config.yaml) from {here}",
    )
