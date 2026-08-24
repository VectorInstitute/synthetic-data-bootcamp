"""Local Qwen-family image editing (distinct from Klein diffusion instruct).

Uses ``QwenImageEditPipeline`` (Qwen-Image-Edit). Same ecosystem as the
Qwen2.5-VL *judge*, but this model outputs edited pixels — not JSON scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from PIL import Image

from edgecase_synthesis.conditioning import resolve_device
from edgecase_synthesis.generation import _ensure_same_size, _fit_for_diffusion


@dataclass
class VlmLocalEditConfig:
    model_id: str = "Qwen/Qwen-Image-Edit"
    num_inference_steps: int = 20
    true_cfg_scale: float = 4.0
    max_side: int = 768
    negative_prompt: str = "blurry, distorted, cartoon, painting, watermark, text overlay"


_pipe_cache: dict[tuple[str, str], Any] = {}


def _cache_key(model_id: str, device: str) -> tuple[str, str]:
    return model_id, device


def unload_qwen_edit_pipeline(*, model_id: str | None = None, device: str | None = None) -> None:
    """Drop cached pipeline (call before loading Klein / SD pipes on the same GPU)."""
    global _pipe_cache
    if model_id is None and device is None:
        _pipe_cache.clear()
    else:
        keys = [
            k
            for k in _pipe_cache
            if (model_id is None or k[0] == model_id) and (device is None or k[1] == device)
        ]
        for k in keys:
            _pipe_cache.pop(k, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_pipeline(model_id: str, device: torch.device) -> Any:
    key = _cache_key(model_id, str(device))
    if key in _pipe_cache:
        return _pipe_cache[key]

    try:
        from diffusers import QwenImageEditPipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "diffusers with QwenImageEditPipeline is required for vlm_generate_local. "
            "Install the edge-case-image-generation dependency group."
        ) from exc

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    # Spread across GPUs when available (e.g. gpu_l4x2).
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "balanced"
        pipe = QwenImageEditPipeline.from_pretrained(model_id, **kwargs)
    else:
        pipe = QwenImageEditPipeline.from_pretrained(model_id, **kwargs)
        pipe.to(device)

    pipe.set_progress_bar_config(disable=True)
    _pipe_cache[key] = pipe
    return pipe


@torch.inference_mode()
def edit_with_qwen_local(
    image: Image.Image,
    prompt: str,
    *,
    config: VlmLocalEditConfig | None = None,
    device: str | None = None,
    seed: int = 42,
    family: Literal["sd15", "sdxl", "flux"] | str = "sd15",
) -> Image.Image:
    """Run Qwen-Image-Edit on a seed photo + text instruction."""
    cfg = config or VlmLocalEditConfig()
    dev = resolve_device(device)
    original = image.convert("RGB")
    run_image = _fit_for_diffusion(original, max_side=int(cfg.max_side), family=str(family).lower())

    pipe = _get_pipeline(str(cfg.model_id), dev)
    gen_device = "cuda" if dev.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    instruction = str(prompt).strip()
    out = pipe(
        image=run_image,
        prompt=instruction,
        negative_prompt=str(cfg.negative_prompt or ""),
        num_inference_steps=int(cfg.num_inference_steps),
        true_cfg_scale=float(cfg.true_cfg_scale),
        generator=generator,
    )
    edited = out.images[0]
    return _ensure_same_size(edited, original)
