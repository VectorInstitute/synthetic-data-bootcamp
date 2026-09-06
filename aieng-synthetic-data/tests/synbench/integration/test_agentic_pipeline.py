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


@pytest.mark.integration_test
def test_single_agent_pipeline_action_trace(mock_retail_path):
    """Single-agent dialogue records the fixture cancel action trace."""
    domain = load_domain(mock_retail_path)
    task = _load_fixture_task()
    session = SingleToolAgent(domain).run_task(task)

    assert "executor" in session.role_trace
    actions = [action_fingerprint(a) for a in session.agent_actions]
    assert len(actions) > 0, "No actions found in the session"


@pytest.mark.integration_test
def test_multi_agent_pipeline_action_trace(mock_retail_path):
    """Multi-agent dialogue records the fixture cancel action trace."""
    domain = load_domain(mock_retail_path)
    task = _load_fixture_task()
    session = AgentPipeline(domain).run_task(task)

    assert "planner" in session.role_trace
    assert "executor" in session.role_trace
    assert "critic" in session.role_trace
    actions = [action_fingerprint(a) for a in session.agent_actions]
    assert len(actions) > 0, "No actions found in the session"
