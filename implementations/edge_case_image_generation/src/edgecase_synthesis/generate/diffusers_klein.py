"""Diffusers helpers with clear errors for FLUX.2 Klein."""

from __future__ import annotations

from typing import Any

import torch


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


def klein_device_map(device: torch.device | str | None) -> str:
    """Map a torch device to a diffusers ``device_map`` target."""
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        if device in {"cuda", "cuda:"}:
            return "cuda:0"
        return device
    if device.type != "cuda":
        return "cpu"
    idx = 0 if device.index is None else int(device.index)
    return f"cuda:{idx}"


def from_pretrained_klein(
    pipeline_cls: Any,
    model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    prefer_device_map: bool = False,
) -> Any:
    """Load FLUX.2 Klein.

    Default = classic ``from_pretrained`` (fast single-L4 path used before dual-GPU
    experiments). Set ``prefer_device_map=True`` only for multi-GPU workers.
    """
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    if prefer_device_map:
        device_map = klein_device_map(device)
        attempts.extend(
            [
                {"dtype": dtype, "device_map": device_map},
                {"torch_dtype": dtype, "device_map": device_map},
            ]
        )
    attempts.extend(
        [
            {"dtype": dtype},
            {"torch_dtype": dtype},
            {"dtype": dtype, "low_cpu_mem_usage": False},
            {"torch_dtype": dtype, "low_cpu_mem_usage": False},
        ]
    )
    if not prefer_device_map:
        device_map = klein_device_map(device)
        attempts.extend(
            [
                {"dtype": dtype, "device_map": device_map},
                {"torch_dtype": dtype, "device_map": device_map},
            ]
        )
    for kwargs in attempts:
        try:
            return pipeline_cls.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
            continue
    raise RuntimeError(
        "Failed to load FLUX.2 Klein pipeline.\n" + "\n".join(errors)
    )


def configure_klein_pipe(
    pipe: Any,
    *,
    device: torch.device | str | None = None,
    disable_progress: bool = False,
    use_cpu_offload: bool = True,
) -> Any:
    """Progress / slicing + optional cpu_offload (the fast single-GPU recipe)."""
    pipe.set_progress_bar_config(disable=bool(disable_progress))
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if not use_cpu_offload or not hasattr(pipe, "enable_model_cpu_offload"):
        return pipe
    gpu_id = 0
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
        gpu_id = int(device.index)
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_id = int(device.split(":")[1])
        except (IndexError, ValueError):
            gpu_id = 0
    try:
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    except NotImplementedError:
        # Tied/meta weights on some transformers builds — pipeline may already
        # be on device via device_map fallback in from_pretrained_klein.
        pass
    return pipe
