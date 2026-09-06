"""Tests that FSM validation accepts a well-formed action sequence."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.fsm.validator import validate_actions_against_fsm
from aieng.syn_data.synbench.schemas.tasks import Task


def test_valid_cancel_path(mock_retail_path):
    """Lookup followed by cancel matches the cancel task type's path."""
    domain = load_domain(mock_retail_path)
    draft = Task.model_validate(
        {
            "id": "t",
            "task_type": "cancel",
            "evaluation_criteria": {
                "actions": [
                    {"name": "get_order", "arguments": {"order_id": "ord_1001"}},
                    {"name": "cancel_order", "arguments": {"order_id": "ord_1001"}},
                ]
            },
        }
    )
    validate_actions_against_fsm(
        domain, draft.task_type, draft.evaluation_criteria.actions
    )
