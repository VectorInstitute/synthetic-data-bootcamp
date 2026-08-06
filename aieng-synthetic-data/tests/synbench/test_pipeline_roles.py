"""Tests for role wiring in the multi-agent pipeline."""

from aieng.syn_data.synbench.agents.pipeline import AgentPipeline
from aieng.syn_data.synbench.domain.loader import load_domain


def test_pipeline_role_order(mock_retail_path):
    """The default pipeline includes planner and executor and solves the task."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    pipeline = AgentPipeline(domain)
    score = pipeline.run_and_score_task(task)
    assert "planner" in pipeline.roles
    assert "executor" in pipeline.roles
    assert score.reward == 1.0


def test_pipeline_first_user_message_is_initial(mock_retail_path):
    """The dialogue opens with the task's authored initial message."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    pipeline = AgentPipeline(domain, max_dialogue_turns=2)
    session = pipeline.run_task(task)
    user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert user_msgs[0] == task.user_scenario.initial_message
    assert session.role_trace[0] == "planner" or "executor" in session.role_trace
