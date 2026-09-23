"""LLM calls used during task generation."""

from __future__ import annotations

from typing import Any

from aieng.syn_data.synbench.llm.client import get_client
from aieng.syn_data.synbench.llm.config import get_generator_model


__all__ = ["call_llm_json"]


def call_llm_json(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Ask the LLM for a JSON task draft."""
    generator_model = get_generator_model()
    client = get_client(model=generator_model)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You output only valid JSON for synthetic benchmark tasks.",
        },
        {"role": "user", "content": prompt},
    ]
    return client.complete_json(messages)
