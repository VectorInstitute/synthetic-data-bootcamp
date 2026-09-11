"""Provide shared helpers for cloud vision and image API calls."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal

from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image


ApiProvider = Literal["gemini", "openai", "vector_proxy"]

VECTOR_PROXY_BASE_URL = "https://proxy.vectorinstitute.ai/v1"

# Friendly names → API model IDs (vision chat → text). Not image-generation IDs.
JUDGE_MODEL_ALIASES: dict[str, str] = {
    "gemini 3 flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini 3.1 flash": "gemini-3.1-flash-preview",
    "gemini-3.1-flash": "gemini-3.1-flash-preview",
    "gemini 3.5 flash": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
}


def resolve_judge_model(name: str) -> str:
    """Resolve the configured judge model."""
    raw = str(name or "").strip()
    spaced = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    return JUDGE_MODEL_ALIASES.get(spaced, raw)


def infer_api_provider(model: str, *, api_base_url: str | None = None) -> ApiProvider:
    """Infer the API provider for a model."""
    if api_base_url:
        return "vector_proxy"
    mid = model.lower()
    if mid.startswith("gpt") or "openai" in mid:
        return "openai"
    return "gemini"


def _first_env(*names: str) -> str | None:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return None


def vector_api_key(explicit: str | None = None) -> str:
    """Baseline Vector proxy key (``OPENAI_API_KEY`` / ``VECTOR_PROXY_API_KEY``)."""
    key = explicit or _first_env("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY")
    if not key:
        raise EnvironmentError(
            "Missing Vector API key. Set OPENAI_API_KEY in "
            "implementations/edge_case_image_generation/.env (see .env.example).",
        )
    return key


def proxy_api_key(explicit: str | None = None) -> str:
    """Alias for :func:`vector_api_key` (kept for older call sites)."""
    return vector_api_key(explicit)


def resolve_api_key(
    explicit: str | None = None,
    *,
    role: str | None = None,
) -> str:
    """Resolve an API key: explicit → role env → Vector baseline.

    Role env names (examples): ``JUDGE_API_KEY``, ``VLM_API_KEY``.
    """
    if explicit:
        return explicit
    if role:
        role_key = _first_env(f"{role.upper()}_API_KEY")
        if role_key:
            return role_key
    return vector_api_key()


def resolve_api_base_url(
    explicit: str | None = None,
    *,
    role: str | None = None,
) -> str:
    """Resolve an OpenAI-compatible base URL: explicit → role env → Vector proxy."""
    if explicit:
        return str(explicit)
    if role:
        role_url = _first_env(f"{role.upper()}_API_BASE_URL")
        if role_url:
            return role_url
    return resolve_proxy_base_url()


def resolve_proxy_base_url(explicit: str | None = None) -> str:
    """Resolve the proxy API base URL."""
    return (
        explicit
        or os.environ.get("VECTOR_PROXY_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or VECTOR_PROXY_BASE_URL
    )


def gemini_api_key(explicit: str | None = None) -> str:
    """Direct Google key; falls back to Vector key only if no Gemini env is set."""
    key = explicit or _first_env(
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    )
    if key:
        return key
    # Last resort: allow workshop Vector key only when caller forgot Gemini.
    try:
        return vector_api_key()
    except EnvironmentError as exc:
        raise EnvironmentError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
            "or OPENAI_API_KEY for the Vector proxy path.",
        ) from exc


def openai_api_key(explicit: str | None = None) -> str:
    """Return the configured OpenAI API key."""
    return vector_api_key(explicit)


def make_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    role: str | None = None,
) -> Any:
    """Create openai client."""
    try:
        pass
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openai is required for API judge / Vector proxy. Install: uv sync --group edge-case-image-generation",
        ) from exc
    key = resolve_api_key(api_key, role=role)
    url = resolve_api_base_url(base_url, role=role) if (base_url or role) else None
    if url:
        return OpenAI(api_key=key, base_url=url)
    return OpenAI(api_key=key)


def pil_to_png_bytes(image: Image.Image, *, max_side: int | None = 1024) -> bytes:
    """Encode a PIL image as PNG bytes."""
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


def pil_to_b64(image: Image.Image, *, max_side: int | None = 1024) -> str:
    """Encode a PIL image as base64 PNG data."""
    return base64.b64encode(pil_to_png_bytes(image, max_side=max_side)).decode("ascii")


def vision_chat(
    user_text: str,
    image: Image.Image | list[Image.Image],
    *,
    model: str,
    provider: ApiProvider | None = None,
    api_key: str | None = None,
    api_base_url: str | None = None,
    max_side: int = 1024,
) -> str:
    """Multimodal chat: one or more RGB images + text → assistant text."""
    images = [image] if isinstance(image, Image.Image) else list(image)
    if not images:
        raise ValueError("vision_chat requires at least one image")
    model = resolve_judge_model(model)
    provider = provider or infer_api_provider(model, api_base_url=api_base_url)
    if provider == "vector_proxy":
        return _vision_chat_openai(
            user_text,
            images,
            model=model,
            api_key=api_key,
            api_base_url=resolve_api_base_url(api_base_url, role="JUDGE"),
            max_side=max_side,
            role="JUDGE",
        )
    if provider == "openai":
        return _vision_chat_openai(
            user_text,
            images,
            model=model,
            api_key=api_key,
            api_base_url=api_base_url,
            max_side=max_side,
            role="JUDGE",
        )
    return _vision_chat_gemini(user_text, images, model=model, api_key=api_key, max_side=max_side)


def _vision_chat_gemini(
    user_text: str,
    images: list[Image.Image],
    *,
    model: str,
    api_key: str | None,
    max_side: int,
) -> str:
    try:
        pass
        pass
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai is required for Gemini API judge. Install: uv sync --group edge-case-image-generation",
        ) from exc

    client = genai.Client(api_key=gemini_api_key(api_key))
    parts: list[Any] = [
        types.Part.from_bytes(data=pil_to_png_bytes(img, max_side=max_side), mime_type="image/png") for img in images
    ]
    parts.append(types.Part.from_text(text=user_text))
    response = client.models.generate_content(model=model, contents=parts)
    text = getattr(response, "text", None)
    if text:
        return str(text)
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text)
    raise RuntimeError(f"Gemini model {model!r} returned no text for judge prompt.")


def _vision_chat_openai(
    user_text: str,
    images: list[Image.Image],
    *,
    model: str,
    api_key: str | None,
    api_base_url: str | None = None,
    max_side: int,
    role: str | None = None,
) -> str:
    client = make_openai_client(api_key=api_key, base_url=api_base_url, role=role)
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img in images:
        b64 = pil_to_b64(img, max_side=max_side)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return str(getattr(response.choices[0].message, "content", None) or "")
