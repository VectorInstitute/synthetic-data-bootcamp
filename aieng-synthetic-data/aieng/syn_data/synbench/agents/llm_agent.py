"""Serialization helpers shared by agent implementations."""

from __future__ import annotations

import json

from aieng.syn_data.synbench.schemas.actions import Action


def actions_to_json(actions: list[Action]) -> str:
    """Serialize a list of tool-call actions to a JSON string."""
    return json.dumps([a.model_dump() for a in actions])
