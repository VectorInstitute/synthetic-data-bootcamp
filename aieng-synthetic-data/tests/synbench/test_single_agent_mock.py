"""Tests for the single-agent runner against the scripted mock client."""

from aieng.syn_data.synbench.agents.loop import ToolCallingLoop
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.single import SingleToolAgent
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.llm.client import LLMResponse
from aieng.syn_data.synbench.llm.mock_client import MockLLMClient


def test_single_agent_scores_seed_cancel(mock_retail_path):
    """The scripted agent earns full reward on the seed cancel task."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    agent = SingleToolAgent(domain)
    score = agent.run_and_score_task(task)
    assert score.reward == 1.0
    assert score.db_reward == 1.0


def test_single_agent_first_user_message_is_initial(mock_retail_path):
    """The dialogue opens with the task's authored initial message."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    agent = SingleToolAgent(domain, max_dialogue_turns=2)
    session = agent.run_task(task)
    user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert user_msgs[0] == task.user_scenario.initial_message
    assert "user_sim" in session.role_trace
    assert "single" in session.role_trace


def test_single_agent_stops_when_user_sim_done(mock_retail_path):
    """Mock user_sim returns [[DONE]] after the first agent reply."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    agent = SingleToolAgent(domain, max_dialogue_turns=5)
    session = agent.run_task(task)
    user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0] == task.user_scenario.initial_message


class _FollowUpUserSimClient(MockLLMClient):
    """Executor follows oracle; user_sim emits one follow-up then DONE."""

    def complete(self, messages, tools=None, *, json_mode=False):
        role = self._role_from_messages(messages)
        if role == "user_sim":
            idx = self._turn_index(self._session_key(messages), "user_sim")
            if idx == 0:
                return LLMResponse(content="Also, how long will shipping take?")
            return LLMResponse(content="[[DONE]]")
        return super().complete(messages, tools=tools, json_mode=json_mode)


def test_single_agent_second_dialogue_turn_appends_follow_up(mock_retail_path):
    """A simulator follow-up becomes a second user message in the transcript."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[0]
    client = _FollowUpUserSimClient()
    agent = SingleToolAgent(domain, max_dialogue_turns=3, client=client)
    session = agent.run_task(task)
    user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert user_msgs[0] == task.user_scenario.initial_message
    assert "shipping" in user_msgs[1].lower()
    assert len(user_msgs) == 2


def test_loop_empty_initial_user_message_does_not_reappend(mock_retail_path):
    """Passing ``""`` keeps the loop from re-seeding the initial message."""
    domain = load_domain(mock_retail_path)
    task = domain.seed_tasks[1]
    client = MockLLMClient()
    session = AgentSession(domain=domain, task=task)
    session.messages.append(
        {"role": "user", "content": task.user_scenario.initial_message}
    )
    loop = ToolCallingLoop(domain, client=client)
    session = loop.run(task, session=session, initial_user_message="")
    user_msgs = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert user_msgs.count(task.user_scenario.initial_message) == 1
    # Critic-style retry must not fall back to initial_message when "" is passed.
    before = len(session.messages)
    session = loop.run(task, session=session, initial_user_message="")
    user_msgs_after = [m for m in session.messages if m.get("role") == "user"]
    assert len(user_msgs_after) == 1
    assert len(session.messages) >= before
