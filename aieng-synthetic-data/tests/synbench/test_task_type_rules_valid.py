"""Tests that task-type validation accepts write usage matching the task type."""

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import Task
from aieng.syn_data.synbench.verification.task_type_checks import (
    validate_task_type_actions,
)


def test_valid_cancel_with_write(mock_retail_path):
    """A cancel oracle that includes a write tool is accepted."""
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
    validate_task_type_actions(
        domain, draft.task_type, draft.evaluation_criteria.actions
    )


def test_write_only_cancel_accepted(mock_retail_path):
    """Order of lookups is not checked; a lone write is enough for cancel."""
    domain = load_domain(mock_retail_path)
    validate_task_type_actions(
        domain,
        "cancel",
        [Action(name="cancel_order", arguments={"order_id": "ord_1001"})],
    )


def test_inquiry_read_only_accepted(mock_retail_path):
    """Inquiry with only read tools is accepted."""
    domain = load_domain(mock_retail_path)
    validate_task_type_actions(
        domain,
        "inquiry",
        [Action(name="get_order", arguments={"order_id": "ord_1001"})],
    )
