"""Conversion between domain tool specs and the OpenAI tool schema."""

from __future__ import annotations

from typing import Any

from aieng.syn_data.synbench.schemas.tools import ToolSpec


def tool_specs_to_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert domain ToolSpec list to OpenAI tools format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters or {"type": "object", "properties": {}},
            },
        }
        for spec in tools
    ]
