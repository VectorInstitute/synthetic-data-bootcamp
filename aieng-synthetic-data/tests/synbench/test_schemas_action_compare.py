"""Tests for action normalization and fingerprint comparison."""

from aieng.syn_data.synbench.schemas.actions import (
    Action,
    action_fingerprint,
    normalize_action,
)


def test_normalize_sorts_dict_keys():
    """Argument key order does not affect an action's fingerprint."""
    a = Action(name="get_order", arguments={"order_id": "ord_1", "extra": "x"})
    b = Action(name="get_order", arguments={"extra": "x", "order_id": "ord_1"})
    assert action_fingerprint(normalize_action(a)) == action_fingerprint(
        normalize_action(b)
    )
