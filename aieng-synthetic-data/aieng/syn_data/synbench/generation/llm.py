"""LLM calls used during task generation, with a mock path for CI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aieng.syn_data.synbench.llm.client import get_client
from aieng.syn_data.synbench.llm.config import get_model, is_mock_llm
from aieng.syn_data.synbench.schemas.tasks import Task


# Re-export for backward compatibility
__all__ = ["is_mock_llm", "get_model", "load_mock_response", "call_llm_json"]


def load_mock_response(name: str = "draft_cancel.json") -> Task:
    """Load a recorded draft task from the LLM response fixtures."""
    fixtures = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm_responses"
    )
    path = fixtures / name
    with open(path) as f:
        data = json.load(f)
    return Task.model_validate(data)


def call_llm_json(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Ask the LLM for a JSON task draft, or return the mock fixture in mock mode."""
    if is_mock_llm():
        draft = load_mock_response()
        return draft.model_dump()

    client = get_client(model=model or get_model())
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You output only valid JSON for synthetic benchmark tasks.",
        },
        {"role": "user", "content": prompt},
    ]
    return client.complete_json(messages)
