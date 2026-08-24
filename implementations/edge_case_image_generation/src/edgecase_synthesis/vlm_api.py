"""Shared helpers for cloud vision / image API calls (Gemini + OpenAI)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal

from PIL import Image

ApiProvider = Literal["gemini", "openai"]

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
    "gemini 3.5 flash": "gemini-3.1-flash-preview",
    "gemini-3.5-flash": "gemini-3.1-flash-preview",
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


def infer_api_provider(model: str) -> ApiProvider:
    mid = model.lower()
    if mid.startswith("gpt") or "openai" in mid:
        return "openai"
    return "gemini"


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
    max_side: int = 1024,
) -> str:
    """Multimodal chat: one RGB image + text → assistant text."""
    model = resolve_judge_model(model)
    provider = provider or infer_api_provider(model)
    if provider == "gemini":
        return _vision_chat_gemini(user_text, image, model=model, api_key=api_key, max_side=max_side)
    return _vision_chat_openai(user_text, image, model=model, api_key=api_key, max_side=max_side)


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
            "google-genai is required for Gemini API judge. Install: uv sync --extra vlm"
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
    max_side: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openai is required for OpenAI API judge. Install: uv sync --extra vlm"
        ) from exc

    client = OpenAI(api_key=openai_api_key(api_key))
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
