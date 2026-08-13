"""Tool-call action schema and deterministic fingerprinting helpers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """Tool call step in an oracle or agent trajectory (τ-inspired)."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("arguments must be a dict")
        return v


def normalize_action(action: Action) -> Action:
    """Stable normalization for hashing and comparison."""

    def _normalize_value(val: Any) -> Any:
        if isinstance(val, dict):
            # Sort the dictionary keys to ensure consistent ordering
            # of arguments
            return {k: _normalize_value(val[k]) for k in sorted(val)}
        if isinstance(val, list):
            return [_normalize_value(x) for x in val]
        return val

    args = _normalize_value(action.arguments)
    return Action(name=action.name, arguments=args)


def action_fingerprint(action: Action) -> str:
    """Deterministic fingerprint for an action."""
    norm = normalize_action(action)
    payload = {"name": norm.name, "arguments": norm.arguments}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def actions_fingerprint(actions: list[Action]) -> str:
    """Deterministic fingerprint for an ordered action sequence."""
    return "|".join(action_fingerprint(a) for a in actions)
