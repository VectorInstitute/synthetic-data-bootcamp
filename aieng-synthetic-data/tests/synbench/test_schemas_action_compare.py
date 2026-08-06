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


def test_coerce_numeric_string():
    """Digit-only strings normalize to integers."""
    a = Action(name="t", arguments={"n": "42"})
    norm = normalize_action(a)
    assert norm.arguments["n"] == 42


def test_float_int_coercion():
    """Whole-number floats normalize to integers."""
    a = Action(name="t", arguments={"x": 1.0})
    norm = normalize_action(a)
    assert norm.arguments["x"] == 1
