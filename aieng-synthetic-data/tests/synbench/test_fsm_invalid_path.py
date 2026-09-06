"""Tests that FSM validation rejects out-of-order action sequences."""

import pytest

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.fsm.validator import (
    FSMValidationError,
    validate_actions_against_fsm,
)
from aieng.syn_data.synbench.schemas.actions import Action


def test_mutate_before_lookup_rejected(mock_retail_path):
    """Cancelling before looking the order up violates the cancel path."""
    domain = load_domain(mock_retail_path)
    actions = [
        Action(name="cancel_order", arguments={"order_id": "ord_1001"}),
        Action(name="get_order", arguments={"order_id": "ord_1001"}),
    ]
    with pytest.raises(FSMValidationError):
        validate_actions_against_fsm(domain, "cancel", actions)
