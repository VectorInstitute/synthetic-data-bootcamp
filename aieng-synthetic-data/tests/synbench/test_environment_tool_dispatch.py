"""Tests for dispatching tool calls against a domain environment."""

import pytest

from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.environment.core import Environment, ToolDispatchError
from aieng.syn_data.synbench.schemas.actions import Action


def test_each_mock_tool(mock_retail_path):
    """Every declared mock_retail read tool dispatches successfully."""
    domain = load_domain(mock_retail_path)
    env = Environment(domain)
    result = env.dispatch(Action(name="find_user_id", arguments={"name": "Alice Chen"}))
    assert result["user_id"] == "user_alice"
    env.dispatch(Action(name="get_order", arguments={"order_id": "ord_1001"}))
    env.dispatch(Action(name="list_orders", arguments={"user_id": "user_alice"}))


def test_unknown_tool_raises(mock_retail_path):
    """Calling a tool the domain does not declare is rejected."""
    domain = load_domain(mock_retail_path)
    env = Environment(domain)
    with pytest.raises(ToolDispatchError, match="Unknown tool"):
        env.dispatch(Action(name="no_such_tool", arguments={}))
