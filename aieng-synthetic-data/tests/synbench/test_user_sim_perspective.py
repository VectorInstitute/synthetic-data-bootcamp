"""Tests that the user simulator sees the transcript from the customer's side."""

from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.agents.user_sim import AGENT_TURN_NUDGE, UserSimulator
from aieng.syn_data.synbench.domain.loader import load_domain
from aieng.syn_data.synbench.llm.client import LLMResponse


class _CaptureClient:
    """Records the message list handed to the LLM and replies with fixed text."""

    def __init__(self, reply: str = "It is ord_1001."):
        self.reply = reply
        self.seen: list[dict] = []

    def complete(self, messages, tools=None, *, json_mode=False):
        """Record ``messages`` and return the canned reply."""
        self.seen = messages
        return LLMResponse(content=self.reply)


def _session(domain):
    """Build a session whose transcript already has one full agent exchange."""
    task = domain.seed_tasks[0]
    session = AgentSession(domain=domain, task=task)
    session.messages = [
        {"role": "system", "content": "agent system prompt"},
        {"role": "user", "content": task.user_scenario.initial_message},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": '{"order_id": "ord_1001"}',
        },
        {"role": "assistant", "content": "Which order ID should I look up?"},
    ]
    return session


def test_transcript_is_flipped_to_customer_perspective(mock_retail_path):
    """Customer turns become assistant turns and agent turns become user turns."""
    domain = load_domain(mock_retail_path)
    session = _session(domain)
    client = _CaptureClient()

    UserSimulator(client).respond(session, "Which order ID should I look up?")

    assert client.seen[0]["role"] == "system"
    turns = client.seen[1:]
    # Customer speaks as assistant, the agent as user.
    assert turns == [
        {"role": "assistant", "content": session.task.user_scenario.initial_message},
        {"role": "user", "content": "Which order ID should I look up?"},
    ]


def test_transcript_drops_tool_and_system_messages(mock_retail_path):
    """Internal tool and system messages are hidden from the customer."""
    domain = load_domain(mock_retail_path)
    session = _session(domain)
    client = _CaptureClient()

    UserSimulator(client).respond(session)

    roles = [m["role"] for m in client.seen[1:]]
    assert "tool" not in roles
    assert "system" not in roles
    assert all('"order_id"' not in m["content"] for m in client.seen[1:])


def test_transcript_ends_on_an_agent_turn(mock_retail_path):
    """No executor text reply must still leave something for the customer to answer."""
    domain = load_domain(mock_retail_path)
    session = AgentSession(domain=domain, task=domain.seed_tasks[0])
    session.messages = [
        {"role": "system", "content": "agent system prompt"},
        {"role": "user", "content": "Where is my order?"},
    ]
    client = _CaptureClient()

    UserSimulator(client).respond(session)

    assert client.seen[-1] == {"role": "user", "content": AGENT_TURN_NUDGE}


def test_last_agent_reply_is_not_duplicated(mock_retail_path):
    """An agent reply already in the transcript is not appended twice."""
    domain = load_domain(mock_retail_path)
    session = _session(domain)
    client = _CaptureClient()

    UserSimulator(client).respond(session, "Which order ID should I look up?")

    agent_turns = [m["content"] for m in client.seen[1:] if m["role"] == "user"]
    assert agent_turns.count("Which order ID should I look up?") == 1


def test_speaker_prefix_is_stripped(mock_retail_path):
    """A leading speaker name is removed from the simulator's reply."""
    domain = load_domain(mock_retail_path)
    session = _session(domain)
    client = _CaptureClient(reply="Alice Chen: it is ord_1001.")

    assert UserSimulator(client).respond(session) == "it is ord_1001."


def test_is_done_covers_blank_and_wrapped_output():
    """Blank replies and any message containing ``[[DONE]]`` end the dialogue."""
    sim = UserSimulator(_CaptureClient())
    assert sim.is_done("")
    assert sim.is_done("   ")
    assert sim.is_done("[[DONE]]")
    assert sim.is_done("[[DONE]] thanks!")
    assert not sim.is_done("It is ord_1001.")
