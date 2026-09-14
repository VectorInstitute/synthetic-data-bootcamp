"""Parsing of OpenAI-compatible assistant messages into ``Action`` objects."""

from __future__ import annotations

import json
from typing import Any

from aieng.syn_data.synbench.schemas.actions import Action


def parse_tool_call(name: str, arguments: str | dict[str, Any]) -> Action:
    """Build an ``Action`` from a tool name and raw JSON or dict arguments."""
    if isinstance(arguments, str):
        args = json.loads(arguments) if arguments else {}
    else:
        args = arguments or {}
    return Action(name=name, arguments=args)


def tool_calls_from_message(message: Any) -> list[Action]:
    """Extract Action list from an OpenAI-compatible assistant message."""
    actions: list[Action] = []
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    ) or []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function", {})
        if fn is None:
            continue
        if isinstance(fn, dict):
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
        else:
            name = getattr(fn, "name", "")
            raw_args = getattr(fn, "arguments", "{}")
        if name:
            actions.append(parse_tool_call(name, raw_args))
    return actions


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """Serialize assistant message for chat history."""
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    ) or ""

    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    ) or []

    out: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None and isinstance(tc, dict):
                fn = tc.get("function", {})
            if isinstance(fn, dict):
                serialized.append(
                    {
                        "id": getattr(tc, "id", None) or tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    }
                )
            else:
                serialized.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(fn, "name", ""),
                            "arguments": getattr(fn, "arguments", "{}"),
                        },
                    }
                )
        out["tool_calls"] = serialized
    return out
