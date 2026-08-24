"""Cloud VLM / image-model generation (Gemini image + OpenAI GPT Image).

This is intentionally separate from FLUX.2-klein / InstructPix2Pix:

* **Klein / instruct** — local *image editors* (diffusion conditioned on a seed photo).
* **VLM image models** — cloud multimodal models that *synthesize* pixels from text
  (and optionally a seed image). Stronger world knowledge; weaker hard locks on the
  original photo geometry.

Plain chat VLMs (``gemini-3-flash``, ``gpt-4o`` text) analyze images but do **not**
return new pixels. Use the ``*-image`` / ``gpt-image-*`` IDs below.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

# Friendly notebook names → API model IDs that can emit images.
VLM_MODEL_ALIASES: dict[str, str] = {
    # Gemini — image-capable (Nano Banana family)
    "gemini 3 flash": "gemini-3.1-flash-image",
    "gemini-3-flash": "gemini-3.1-flash-image",
    "gemini 3.1 flash": "gemini-3.1-flash-image",
    "gemini-3.1-flash": "gemini-3.1-flash-image",
    "gemini 3.1 flash lite": "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-image",
    "gemini 3.1 pro": "gemini-3-pro-image-preview",
    "gemini-3.1-pro": "gemini-3-pro-image-preview",
    "gemini 3 pro": "gemini-3-pro-image-preview",
    "gemini-3-pro": "gemini-3-pro-image-preview",
    "gemini 3.5 flash": "gemini-3.1-flash-image",
    "gemini-3.5-flash": "gemini-3.1-flash-image",
    # Pass-through common official IDs
    "gemini-3.1-flash-image": "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image": "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
    # OpenAI — image models (GPT-4o chat alone does not emit pixels)
    "gpt-4o": "gpt-image-1",
    "gpt4o": "gpt-image-1",
    "gpt-image-1": "gpt-image-1",
    "gpt-image-1.5": "gpt-image-1.5",
    "gpt-image-2": "gpt-image-2",
}

VlmMode = Literal["edit", "generate"]
VlmProvider = Literal["gemini", "openai"]


@dataclass
class VlmGenerateConfig:
    """Knobs for cloud image generation."""

    model: str = "gemini-3.1-flash-image"
    mode: VlmMode = "edit"  # edit = seed+instruction; generate = text-only
    provider: VlmProvider | None = None  # auto from model id if None
    api_key: str | None = None
    aspect_ratio: str | None = None  # Gemini only, e.g. "16:9"
    size: str = "1024x1024"  # OpenAI Images API
    max_side: int = 1024


def resolve_vlm_model(name: str) -> str:
    key = str(name or "").strip().lower().replace("_", "-")
    # Normalize spaces for alias lookup
    spaced = " ".join(str(name or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if spaced in VLM_MODEL_ALIASES:
        return VLM_MODEL_ALIASES[spaced]
    if key in VLM_MODEL_ALIASES:
        return VLM_MODEL_ALIASES[key]
    return str(name).strip()


def infer_provider(model: str) -> VlmProvider:
    mid = model.lower()
    if mid.startswith("gpt") or "openai" in mid:
        return "openai"
    return "gemini"


def _pil_to_png_bytes(image: Image.Image, *, max_side: int | None = None) -> bytes:
    img = image.convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale)))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _gemini_api_key(explicit: str | None = None) -> str:
    key = (
        explicit
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )
    if not key:
        raise EnvironmentError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY)."
        )
    return key


def _openai_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OpenAI API key. Set OPENAI_API_KEY.")
    return key


def generate_with_vlm(
    prompt: str,
    *,
    seed_image: Image.Image | None = None,
    config: VlmGenerateConfig | None = None,
) -> Image.Image:
    """Generate or edit an image via Gemini image / OpenAI GPT Image APIs."""
    cfg = config or VlmGenerateConfig()
    model = resolve_vlm_model(cfg.model)
    provider = cfg.provider or infer_provider(model)
    mode = cfg.mode
    if mode == "edit" and seed_image is None:
        raise ValueError("mode='edit' requires a seed_image")
    if mode == "generate":
        seed_image = None

    if provider == "gemini":
        return _generate_gemini(prompt, seed_image=seed_image, model=model, cfg=cfg)
    return _generate_openai(prompt, seed_image=seed_image, model=model, cfg=cfg)


def _generate_gemini(
    prompt: str,
    *,
    seed_image: Image.Image | None,
    model: str,
    cfg: VlmGenerateConfig,
) -> Image.Image:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai is required for Gemini image generation. "
            "Install with: uv add google-genai"
        ) from exc

    client = genai.Client(api_key=_gemini_api_key(cfg.api_key))
    parts: list[Any] = []
    if seed_image is not None:
        parts.append(
            types.Part.from_bytes(
                data=_pil_to_png_bytes(seed_image, max_side=cfg.max_side),
                mime_type="image/png",
            )
        )
        text = (
            "Edit this street photograph. Keep camera angle, layout, vehicles, "
            "buildings, and lighting as similar as possible. "
            f"Instruction: {prompt.strip()}"
        )
    else:
        text = (
            "Generate a photorealistic street-level dashcam / Mapillary-style photo. "
            f"{prompt.strip()}"
        )
    parts.append(types.Part.from_text(text=text))

    gen_cfg_kwargs: dict[str, Any] = {
        "response_modalities": ["TEXT", "IMAGE"],
    }
    if cfg.aspect_ratio:
        gen_cfg_kwargs["image_config"] = types.ImageConfig(aspect_ratio=cfg.aspect_ratio)

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**gen_cfg_kwargs),
    )
    image = _extract_gemini_image(response)
    if image is None:
        raise RuntimeError(
            f"Gemini model {model!r} returned no image. "
            "Use an image-capable ID (e.g. gemini-3.1-flash-image), not a text-only Flash/Pro chat model."
        )
    return image


def _extract_gemini_image(response: Any) -> Image.Image | None:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if not data:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)
            return _bytes_to_pil(data)
    return None


def _generate_openai(
    prompt: str,
    *,
    seed_image: Image.Image | None,
    model: str,
    cfg: VlmGenerateConfig,
) -> Image.Image:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openai is required for GPT image generation. Install with: uv add openai"
        ) from exc

    client = OpenAI(api_key=_openai_api_key(cfg.api_key))
    if seed_image is not None:
        # Image edit: seed + instruction.
        buf = io.BytesIO(_pil_to_png_bytes(seed_image, max_side=cfg.max_side))
        buf.name = "seed.png"
        result = client.images.edit(
            model=model,
            image=buf,
            prompt=(
                "Edit this street photograph. Keep camera angle, layout, vehicles, "
                f"buildings, and lighting similar. Instruction: {prompt.strip()}"
            ),
            size=cfg.size,  # type: ignore[arg-type]
        )
    else:
        result = client.images.generate(
            model=model,
            prompt=(
                "Photorealistic street-level dashcam / Mapillary-style photo. "
                f"{prompt.strip()}"
            ),
            size=cfg.size,  # type: ignore[arg-type]
        )

    item = result.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return _bytes_to_pil(base64.b64decode(b64))
    url = getattr(item, "url", None)
    if url:
        import urllib.request

        with urllib.request.urlopen(url) as resp:  # noqa: S310 — API-returned HTTPS URL
            return _bytes_to_pil(resp.read())
    raise RuntimeError(f"OpenAI model {model!r} returned no image data.")
