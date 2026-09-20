"""Tests that task-type validation rejects write usage the task type forbids."""

import pytest

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.verification.task_type_checks import (
    TaskTypeValidationError,
    validate_task_type_actions,
)


def test_write_on_read_only_task_rejected(mock_retail_path):
    """Inquiry forbids write tools even if a lookup is also present."""
    domain = load_domain(mock_retail_path)
    actions = [
        Action(name="get_order", arguments={"order_id": "ord_1001"}),
        Action(name="cancel_order", arguments={"order_id": "ord_1001"}),
    ]
    with pytest.raises(TaskTypeValidationError, match="forbids write tools"):
        validate_task_type_actions(domain, "inquiry", actions)


def test_write_task_without_write_rejected(mock_retail_path):
    """Cancel requires at least one write tool."""
    domain = load_domain(mock_retail_path)
    actions = [
        Action(name="get_order", arguments={"order_id": "ord_1001"}),
    ]
    with pytest.raises(
        TaskTypeValidationError, match="requires at least one write tool"
    ):
        validate_task_type_actions(domain, "cancel", actions)
