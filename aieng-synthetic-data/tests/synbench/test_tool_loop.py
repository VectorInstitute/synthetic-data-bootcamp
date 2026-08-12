"""Tests for the tool-calling loop with a scripted stub client."""

from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.llm.client import LLMResponse
from aieng.syn_data.synbench.schemas.actions import Action
from aieng.syn_data.synbench.schemas.tasks import Task


class _OracleStubClient:
    """Replay oracle tool calls from the task, then a final text reply."""

    def __init__(self, task: Task):
        self._responses: list[LLMResponse] = []
        for action in task.evaluation_criteria.actions:
            self._responses.append(
                LLMResponse(
                    tool_calls=[Action(name=action.name, arguments=action.arguments)]
                )
            )
        comm = task.evaluation_criteria.communicate_info
        self._responses.append(LLMResponse(content=" ".join(comm) if comm else "Done."))
        self._idx = 0

    def complete(self, messages, tools=None, *, json_mode=False):
        if self._idx >= len(self._responses):
            return LLMResponse(content="Done.")
        response = self._responses[self._idx]
        self._idx += 1
        return response


def test_tool_loop_collects_actions(mock_retail_path):
    """The loop records each dispatched tool call in order."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    client = _OracleStubClient(task)
    session = ToolCallingLoop(domain, client=client).run(task)
    assert len(session.agent_actions) == 2
    assert session.agent_actions[0].name == "get_order"
    assert session.agent_actions[1].name == "cancel_order"
    assert any("cancel" in m.lower() for m in session.agent_messages)
