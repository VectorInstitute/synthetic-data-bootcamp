"""Scoring of aborted runs vs recoverable tool errors."""

from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.single import SingleToolAgent
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.evaluation.scoring import (
    failed_execution_score,
    score_agent_run,
    score_trajectory,
)
from aieng.syn_data.synbench.llm.client import LLMResponse
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import Task


class _ScriptedClient:
    """Return a fixed sequence of LLM responses, then a text reply."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._idx = 0

    def complete(self, messages, tools=None, *, json_mode=False):
        if self._idx >= len(self._responses):
            return LLMResponse(content="Done.")
        response = self._responses[self._idx]
        self._idx += 1
        return response

    def complete_json(self, messages):
        raise NotImplementedError


def test_loop_records_tool_error_and_continues(mock_retail_path):
    """A bad lookup is stored on the session and fed back as a tool error."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[Action(name="find_user_id", arguments={"name": "unknown"})]
            ),
            LLMResponse(content="I need your full name."),
        ]
    )
    session = ToolCallingLoop(domain, client=client).run(task)
    assert session.tool_errors
    assert "User not found" in session.tool_errors[0]
    tool_msgs = [
        m.get("content") or "" for m in session.messages if m.get("role") == "tool"
    ]
    assert any("User not found" in content for content in tool_msgs)


def test_failed_execution_score_is_zero_without_hashes():
    """An aborted run is reward 0 and does not claim a predicted DB hash."""
    result = failed_execution_score(["LLM timeout"])
    assert result.reward == 0.0
    assert result.db_reward == 0.0
    assert result.execution_ok is False
    assert result.tool_errors == ["LLM timeout"]
    assert result.predicted_db_hash == ""


def test_score_agent_run_catches_uncaught_exception(mock_retail_path):
    """Exceptions that escape run_task skip replay and zero the reward."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]

    def boom(_task: Task):
        raise RuntimeError("client exploded")

    result = score_agent_run(domain, task, boom)
    assert result.execution_ok is False
    assert result.reward == 0.0
    assert "client exploded" in result.tool_errors[0]


def test_tool_errors_do_not_zero_a_successful_outcome(mock_retail_path):
    """A recovered trajectory still scores from replay, with errors attached."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    result = score_trajectory(
        domain,
        task,
        task.evaluation_criteria.actions,
        task.evaluation_criteria.communicate_info,
        tool_errors=["User not found for name: unknown"],
    )
    assert result.execution_ok is True
    assert result.reward == 1.0
    assert result.tool_errors == ["User not found for name: unknown"]


def test_single_agent_run_and_score_aborted_task(mock_retail_path):
    """SingleToolAgent.run_and_score_task maps a crashing run to reward 0."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    agent = SingleToolAgent(domain)

    def boom(_task: Task):
        raise RuntimeError("dialogue failed")

    agent.run_task = boom  # type: ignore[method-assign]
    result = agent.run_and_score_task(task)
    assert result.reward == 0.0
    assert result.execution_ok is False
