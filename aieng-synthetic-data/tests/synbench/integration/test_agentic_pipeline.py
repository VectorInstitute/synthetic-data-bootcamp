"""Integration tests for single- and multi-agent task runs."""

import json
from pathlib import Path

import pytest

from aieng.syn_data.synbench.agents.pipeline import AgentPipeline
from aieng.syn_data.synbench.agents.single import SingleToolAgent
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.schemas.actions import action_fingerprint
from aieng.syn_data.synbench.schemas.tasks import Task


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_fixture_task() -> Task:
    with open(FIXTURES / "tasks_valid.json") as f:
        return Task.model_validate(json.load(f))


def _assert_action_trace(session, task: Task) -> None:
    """Assert action trace based on the order of oracle tool actions in the session.

    Live agents may insert extra read-only lookups; require the expected
    actions as an ordered subsequence of ``session.agent_actions``.
    """
    expected = [action_fingerprint(a) for a in task.evaluation_criteria.actions]
    actual = [action_fingerprint(a) for a in session.agent_actions]
    i = 0
    for fp in actual:
        if i < len(expected) and fp == expected[i]:
            i += 1
    assert i == len(expected), (
        f"oracle actions missing from session trace\n"
        f"expected subsequence: {expected}\n"
        f"actual: {actual}"
    )


@pytest.mark.integration
def test_single_agent_pipeline_action_trace(mock_retail_path):
    """Single-agent dialogue records the fixture cancel action trace."""
    domain = load_domain(mock_retail_path)
    task = _load_fixture_task()
    session = SingleToolAgent(domain).run_task(task)

    assert "executor" in session.role_trace
    _assert_action_trace(session, task)


@pytest.mark.integration
def test_multi_agent_pipeline_action_trace(mock_retail_path):
    """Multi-agent dialogue records the fixture cancel action trace."""
    domain = load_domain(mock_retail_path)
    task = _load_fixture_task()
    session = AgentPipeline(domain).run_task(task)

    assert "planner" in session.role_trace
    assert "executor" in session.role_trace
    assert "critic" in session.role_trace
    _assert_action_trace(session, task)
