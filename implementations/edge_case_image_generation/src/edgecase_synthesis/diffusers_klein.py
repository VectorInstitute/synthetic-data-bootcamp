"""Diffusers helpers with clear errors for FLUX.2 Klein."""

from __future__ import annotations

from typing import Any


_KLEIN_INSTALL_HINT = (
    "FLUX.2 Klein requires diffusers>=0.39 (Flux2KleinPipeline). "
    "From the repo root run:\n"
    "  uv sync --dev --group edge-case-image-generation\n"
    "Then restart the notebook kernel. "
    "If sync still leaves an old wheel: uv pip install -U 'diffusers>=0.39'"
)


def diffusers_version() -> str:
    try:
        import diffusers

        return str(getattr(diffusers, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "not-installed"


def import_flux2_klein_pipeline() -> Any:
    try:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            f"{_KLEIN_INSTALL_HINT}\n(Currently installed diffusers={diffusers_version()}.)"
        ) from exc


def import_flux2_klein_inpaint_pipeline() -> Any:
    try:
        from diffusers import Flux2KleinInpaintPipeline

        return Flux2KleinInpaintPipeline
    except ImportError as exc:
        raise ImportError(
            f"{_KLEIN_INSTALL_HINT}\n(Currently installed diffusers={diffusers_version()}.)"
        ) from exc


def assert_klein_available() -> None:
    """Fail fast before loading dual-GPU stacks / long batch runs."""
    import_flux2_klein_pipeline()
