"""Tests for the tool-calling loop and the mock client's scripted turns."""

import os

from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.llm.mock_client import MockLLMClient


def test_tool_loop_collects_actions(mock_retail_path):
    """The loop records each dispatched tool call in order."""
    os.environ["MOCK_LLM"] = "1"
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    client = MockLLMClient()
    session = ToolCallingLoop(domain, client=client).run(task)
    assert len(session.agent_actions) == 2
    assert session.agent_actions[0].name == "get_order"
    assert session.agent_actions[1].name == "cancel_order"
    assert any("cancel" in m.lower() for m in session.agent_messages)


def test_mock_turns_from_task(mock_retail_path):
    """Scripted turns are derived from the task's oracle actions."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    turns = MockLLMClient.turns_from_task(task)
    assert len(turns) == 2
    assert turns[0]["tool_calls"][0]["name"] == "get_order"


def test_mock_ensure_task_does_not_reset_mid_run(mock_retail_path):
    """``ensure_task`` preserves turn counters while ``register_task`` resets them."""
    client = MockLLMClient()
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    turns = MockLLMClient.turns_from_task(task)
    client.register_task(task.id, turns)
    # Advance executor turn counter once.
    client._sessions[f"{task.id}:executor"] = 1
    client.ensure_task(task.id, turns)
    assert client._sessions.get(f"{task.id}:executor") == 1
    # Fresh register still resets.
    client.register_task(task.id, turns)
    assert f"{task.id}:executor" not in client._sessions
