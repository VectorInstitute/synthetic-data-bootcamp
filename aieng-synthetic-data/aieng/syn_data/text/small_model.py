"""Client factory for the small model used in baseline and fine-tuned evaluation."""

from __future__ import annotations

import logging
import os

from aieng.syn_data.text.clients import LLMSettings, OpenAICompatibleClient


logger = logging.getLogger(__name__)


def create_small_model_client(
    *,
    default_model: str = "qwen2.5:3b-instruct",
) -> OpenAICompatibleClient:
    """Create a client for the small model under evaluation.

    Configure via environment variables:

    - ``SMALL_MODEL_API_KEY`` (falls back to ``OPENAI_API_KEY``)
    - ``SMALL_MODEL_NAME`` (falls back to ``OPENAI_MODEL`` or ``default_model``)
    - ``SMALL_MODEL_BASE_URL`` (falls back to ``OPENAI_BASE_URL``)
    """
    api_key = os.getenv("SMALL_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        msg = "Set SMALL_MODEL_API_KEY or OPENAI_API_KEY for small-model inference."
        raise ValueError(msg)

    settings = LLMSettings(
        api_key=api_key,
        model=(
            os.getenv("SMALL_MODEL_NAME") or os.getenv("OPENAI_MODEL") or default_model
        ),
        base_url=(
           os.getenv("SMALL_MODEL_BASE_URL")
           or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
       ).rstrip("/"), 
    )
    logger.info("Creating small model client for model: %s", settings.model)
    return OpenAICompatibleClient(settings)
