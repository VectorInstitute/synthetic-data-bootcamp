"""Cloud image generation / edit (disabled by default on Vector chat proxy).

Local Klein / inpaint / ControlNet do production edits. This module is the optional
NB1.5 ``vlm_generate_api`` column — keep ``generation.vlm_api_enabled=false`` until
the API key has an *image-capable* model (chat VLMs only return text).
"""

from __future__ import annotations

import base64
import io
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from edgecase_synthesis.vlm_api import (
    gemini_api_key,
    make_openai_client,
    pil_to_b64,
    pil_to_png_bytes,
    resolve_proxy_base_url,
)

# Only IDs that can emit pixels. Chat models (gemini-3-flash-preview, gpt-4o) are judge-only.
VLM_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.1-flash-image": "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image": "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "gpt-image-1": "gpt-image-1",
    "gpt-image-1.5": "gpt-image-1.5",
}

VlmMode = Literal["edit", "generate"]
VlmProvider = Literal["gemini", "openai", "vector_proxy"]

_VLM_API_DISABLED_MSG = (
    "vlm_generate_api is disabled (generation.vlm_api_enabled=false). "
    "Vector proxy models are chat/vision — use them for the API judge, not image edit. "
    "Re-enable when an image-capable model ID is available on the key."
)


@dataclass
class VlmGenerateConfig:
    model: str = "gemini-3.1-flash-image"
    mode: VlmMode = "edit"
    provider: VlmProvider | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    aspect_ratio: str | None = None
    size: str = "1024x1024"
    max_side: int = 1024


def require_vlm_api_enabled(flag: Any) -> None:
    """Raise unless ``generation.vlm_api_enabled`` is truthy."""
    if flag is True:
        return
    if isinstance(flag, str) and flag.strip().lower() in {"1", "true", "yes", "on"}:
        return
    raise RuntimeError(_VLM_API_DISABLED_MSG)


def resolve_vlm_model(name: str) -> str:
    raw = str(name or "").strip()
    spaced = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    keyed = raw.lower().replace("_", "-")
    return VLM_MODEL_ALIASES.get(spaced) or VLM_MODEL_ALIASES.get(keyed) or raw


def infer_provider(model: str, *, api_base_url: str | None = None) -> VlmProvider:
    if api_base_url:
        return "vector_proxy"
    if model.lower().startswith("gpt") or "openai" in model.lower():
        return "openai"
    return "gemini"


def generate_with_vlm(
    prompt: str,
    *,
    seed_image: Image.Image | None = None,
    config: VlmGenerateConfig | None = None,
) -> Image.Image:
    cfg = config or VlmGenerateConfig()
    model = resolve_vlm_model(cfg.model)
    provider = cfg.provider or infer_provider(model, api_base_url=cfg.api_base_url)
    if cfg.mode == "edit" and seed_image is None:
        raise ValueError("mode='edit' requires a seed_image")
    if cfg.mode == "generate":
        seed_image = None

    if provider == "vector_proxy":
        return _generate_vector_proxy(prompt, seed_image=seed_image, model=model, cfg=cfg)
    if provider == "gemini":
        return _generate_gemini(prompt, seed_image=seed_image, model=model, cfg=cfg)
    return _generate_openai(prompt, seed_image=seed_image, model=model, cfg=cfg)


def _edit_prompt(prompt: str) -> str:
    return (
        "Edit this street photograph. Keep camera angle, layout, vehicles, "
        "buildings, and lighting as similar as possible. "
        f"Instruction: {prompt.strip()}"
    )


def _gen_prompt(prompt: str) -> str:
    return (
        "Generate a photorealistic street-level dashcam / Mapillary-style photo. "
        f"{prompt.strip()}"
    )


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_from_url_or_b64(*, url: str | None = None, b64: str | None = None) -> Image.Image | None:
    if b64:
        raw = b64.split(",", 1)[-1] if b64.startswith("data:") else b64
        return _bytes_to_pil(base64.b64decode(raw))
    if not url:
        return None
    if url.startswith("data:"):
        match = re.search(r"base64,([A-Za-z0-9+/=\s]+)", url)
        return _bytes_to_pil(base64.b64decode(match.group(1).replace("\n", ""))) if match else None
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — API HTTPS URL
        return _bytes_to_pil(resp.read())


def _extract_images_api(result: Any) -> Image.Image:
    data = getattr(result, "data", None) or []
    if not data:
        raise RuntimeError("Images API returned no data.")
    item = data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return _bytes_to_pil(base64.b64decode(b64))
    img = _image_from_url_or_b64(url=getattr(item, "url", None))
    if img is not None:
        return img
    raise RuntimeError("Images API returned no image payload.")


def _extract_chat_image(response: Any) -> Image.Image | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = choices[0].message
    for item in getattr(message, "images", None) or []:
        if isinstance(item, dict):
            url = (item.get("image_url") or {}).get("url") or item.get("url")
            b64 = item.get("b64_json") or item.get("data")
        else:
            url = getattr(getattr(item, "image_url", None), "url", None)
            b64 = getattr(item, "b64_json", None) or getattr(item, "data", None)
        img = _image_from_url_or_b64(url=url, b64=b64)
        if img is not None:
            return img
    content = getattr(message, "content", None)
    if isinstance(content, str):
        match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", content)
        if match:
            return _bytes_to_pil(base64.b64decode(match.group(1).replace("\n", "")))
    return None


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
            "Install: uv sync --group edge-case-image-generation"
        ) from exc

    client = genai.Client(api_key=gemini_api_key(cfg.api_key))
    parts: list[Any] = []
    if seed_image is not None:
        parts.append(
            types.Part.from_bytes(
                data=pil_to_png_bytes(seed_image, max_side=cfg.max_side),
                mime_type="image/png",
            )
        )
        text = _edit_prompt(prompt)
    else:
        text = _gen_prompt(prompt)
    parts.append(types.Part.from_text(text=text))

    gen_kwargs: dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"]}
    if cfg.aspect_ratio:
        gen_kwargs["image_config"] = types.ImageConfig(aspect_ratio=cfg.aspect_ratio)

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**gen_kwargs),
    )
    for cand in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(cand, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if not data:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)
            return _bytes_to_pil(data)
    raise RuntimeError(
        f"Gemini model {model!r} returned no image. Use an image-capable ID "
        "(e.g. gemini-3.1-flash-image), not a chat-only Flash/Pro model."
    )


def _generate_openai(
    prompt: str,
    *,
    seed_image: Image.Image | None,
    model: str,
    cfg: VlmGenerateConfig,
) -> Image.Image:
    client = make_openai_client(api_key=cfg.api_key, base_url=None)
    if seed_image is not None:
        buf = io.BytesIO(pil_to_png_bytes(seed_image, max_side=cfg.max_side))
        buf.name = "seed.png"
        result = client.images.edit(
            model=model,
            image=buf,
            prompt=_edit_prompt(prompt),
            size=cfg.size,  # type: ignore[arg-type]
        )
    else:
        result = client.images.generate(
            model=model,
            prompt=_gen_prompt(prompt),
            size=cfg.size,  # type: ignore[arg-type]
        )
    return _extract_images_api(result)


def _generate_vector_proxy(
    prompt: str,
    *,
    seed_image: Image.Image | None,
    model: str,
    cfg: VlmGenerateConfig,
) -> Image.Image:
    client = make_openai_client(
        api_key=cfg.api_key,
        base_url=resolve_proxy_base_url(cfg.api_base_url),
    )
    text = _edit_prompt(prompt) if seed_image is not None else _gen_prompt(prompt)

    # Prefer chat multimodal when the proxy supports image out; fall back to Images API.
    messages: list[dict[str, Any]]
    if seed_image is not None:
        b64 = pil_to_b64(seed_image, max_side=cfg.max_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": text}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"response_modalities": ["TEXT", "IMAGE"]},
        )
        image = _extract_chat_image(response)
        if image is not None:
            return image
    except Exception:
        pass

    if seed_image is not None:
        buf = io.BytesIO(pil_to_png_bytes(seed_image, max_side=cfg.max_side))
        buf.name = "seed.png"
        try:
            return _extract_images_api(
                client.images.edit(
                    model=model,
                    image=buf,
                    prompt=text,
                    size=cfg.size,  # type: ignore[arg-type]
                    response_format="b64_json",
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Vector proxy model {model!r} returned no edited image. "
                "Need an image-capable model ID from inference.vectorinstitute.ai."
            ) from exc

    return _extract_images_api(
        client.images.generate(
            model=model,
            prompt=text,
            size=cfg.size,  # type: ignore[arg-type]
            response_format="b64_json",
            n=1,
        )
    )
