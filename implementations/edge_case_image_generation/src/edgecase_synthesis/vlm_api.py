"""Shared helpers for cloud vision / image API calls (Gemini + OpenAI)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal

from PIL import Image

ApiProvider = Literal["gemini", "openai", "vector_proxy"]

VECTOR_PROXY_BASE_URL = "https://proxy.vectorinstitute.ai/v1"

# Vision / judge models (text + image in → text out). NOT image-generation IDs.
JUDGE_MODEL_ALIASES: dict[str, str] = {
    "gemini 3 flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini 3.1 flash": "gemini-3.1-flash-preview",
    "gemini-3.1-flash": "gemini-3.1-flash-preview",
    "gemini 3.1 flash lite": "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini 3.1 pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini 3 pro": "gemini-3-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini 3.5 flash": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
}


def resolve_judge_model(name: str) -> str:
    key = str(name or "").strip().lower().replace("_", "-")
    spaced = " ".join(str(name or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if spaced in JUDGE_MODEL_ALIASES:
        return JUDGE_MODEL_ALIASES[spaced]
    if key in JUDGE_MODEL_ALIASES:
        return JUDGE_MODEL_ALIASES[key]
    return str(name).strip()


def infer_api_provider(model: str, *, api_base_url: str | None = None) -> ApiProvider:
    if api_base_url:
        return "vector_proxy"
    mid = model.lower()
    if mid.startswith("gpt") or "openai" in mid:
        return "openai"
    return "gemini"


def default_api_base_url() -> str | None:
    """Vector Institute proxy (OpenAI-compatible). Override via env or config."""
    return (
        os.environ.get("VECTOR_PROXY_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or VECTOR_PROXY_BASE_URL
    )


def proxy_api_key(explicit: str | None = None) -> str:
    """API key for the Vector OpenAI-compatible proxy (``vp_…`` keys)."""
    key = (
        explicit
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VECTOR_PROXY_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise EnvironmentError(
            "Missing API key for Vector proxy. Set OPENAI_API_KEY (or VECTOR_PROXY_API_KEY)."
        )
    return key


def make_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openai is required for API judge / Vector proxy. Install: uv sync --group edge-case-vlm"
        ) from exc
    if base_url:
        return OpenAI(api_key=proxy_api_key(api_key), base_url=base_url)
    return OpenAI(api_key=openai_api_key(api_key))


def gemini_api_key(explicit: str | None = None) -> str:
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


def openai_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OpenAI API key. Set OPENAI_API_KEY.")
    return key


def pil_to_png_bytes(image: Image.Image, *, max_side: int | None = 1024) -> bytes:
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
    return base64.b64encode(pil_to_png_bytes(image, max_side=max_side)).decode("ascii")


def vision_chat(
    user_text: str,
    image: Image.Image,
    *,
    model: str,
    provider: ApiProvider | None = None,
    api_key: str | None = None,
    api_base_url: str | None = None,
    max_side: int = 1024,
) -> str:
    """Multimodal chat: one RGB image + text → assistant text."""
    model = resolve_judge_model(model)
    base_url = api_base_url
    provider = provider or infer_api_provider(model, api_base_url=base_url)
    if provider == "vector_proxy":
        resolved_base = (
            base_url
            or os.environ.get("VECTOR_PROXY_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or VECTOR_PROXY_BASE_URL
        )
        return _vision_chat_openai(
            user_text,
            image,
            model=model,
            api_key=api_key,
            api_base_url=resolved_base,
            max_side=max_side,
        )
    if provider == "openai":
        return _vision_chat_openai(
            user_text,
            image,
            model=model,
            api_key=api_key,
            api_base_url=base_url,
            max_side=max_side,
        )
    return _vision_chat_gemini(user_text, image, model=model, api_key=api_key, max_side=max_side)


def _vision_chat_gemini(
    user_text: str,
    image: Image.Image,
    *,
    model: str,
    api_key: str | None,
    max_side: int,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai is required for Gemini API judge. "
            "Install: uv sync --group edge-case-vlm"
        ) from exc

    client = genai.Client(api_key=gemini_api_key(api_key))
    parts: list[Any] = [
        types.Part.from_bytes(data=pil_to_png_bytes(image, max_side=max_side), mime_type="image/png"),
        types.Part.from_text(text=user_text),
    ]
    response = client.models.generate_content(model=model, contents=parts)
    text = getattr(response, "text", None)
    if text:
        return str(text)
    # Fallback parse candidates.
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text)
    raise RuntimeError(f"Gemini model {model!r} returned no text for judge prompt.")


def _vision_chat_openai(
    user_text: str,
    image: Image.Image,
    *,
    model: str,
    api_key: str | None,
    api_base_url: str | None = None,
    max_side: int,
) -> str:
    client = make_openai_client(api_key=api_key, base_url=api_base_url)
    b64 = pil_to_b64(image, max_side=max_side)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    choice = response.choices[0].message
    content = getattr(choice, "content", None) or ""
    return str(content)
