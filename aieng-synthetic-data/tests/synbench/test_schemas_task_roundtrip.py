"""Tests for Task serialization, defaults, and legacy field handling."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aieng.syn_data.synbench.schemas.tasks import RewardType, Task


FIXTURES = Path(__file__).parent / "fixtures"


def test_task_roundtrip():
    """A task survives a dump/validate round trip unchanged."""
    with open(FIXTURES / "tasks_valid.json") as f:
        raw = json.load(f)
    task = Task.model_validate(raw)
    again = Task.model_validate(json.loads(task.model_dump_json()))
    assert again.id == task.id
    assert len(again.evaluation_criteria.actions) == 2


def test_invalid_reward_basis_rejected():
    """An unknown reward basis value fails validation."""
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "id": "x",
                "evaluation_criteria": {"reward_basis": ["NOT_A_REWARD"]},
            }
        )


def test_default_reward_basis():
    """A minimal task rewards on both DB state and communication."""
    task = Task.model_validate({"id": "minimal"})
    assert RewardType.DB in task.evaluation_criteria.reward_basis
    assert RewardType.COMMUNICATE in task.evaluation_criteria.reward_basis


def test_unknown_evaluation_criteria_fields_ignored():
    """Unrecognized evaluation criteria keys are dropped on load."""
    task = Task.model_validate(
        {
            "id": "x",
            "evaluation_criteria": {
                "nl_assertions": [{"contains": "canceled"}],
                "env_assertions": [{"field": "status", "equals": "pending"}],
            },
        }
    )
    dumped = task.model_dump()
    assert "nl_assertions" not in dumped["evaluation_criteria"]
    assert "env_assertions" not in dumped["evaluation_criteria"]


def test_action_id_in_legacy_json_ignored():
    """A legacy ``action_id`` field is dropped without breaking the load."""
    task = Task.model_validate(
        {
            "id": "x",
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "get_order_1",
                        "name": "get_order",
                        "arguments": {"order_id": "ord_1001"},
                    }
                ],
            },
        }
    )
    action = task.evaluation_criteria.actions[0]
    assert action.name == "get_order"
    assert "action_id" not in action.model_dump()
