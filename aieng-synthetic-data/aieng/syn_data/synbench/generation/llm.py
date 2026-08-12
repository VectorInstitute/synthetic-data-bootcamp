"""LLM calls used during task generation."""

from __future__ import annotations

from typing import Any

from aieng.syn_data.synbench.llm.client import get_client
from aieng.syn_data.synbench.llm.config import get_model


__all__ = ["get_model", "call_llm_json"]


def call_llm_json(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Ask the LLM for a JSON task draft."""
    client = get_client(model=model or get_model())
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You output only valid JSON for synthetic benchmark tasks.",
        },
        {"role": "user", "content": prompt},
    ]
    return client.complete_json(messages)
