"""Tool-calling loop that drives one agent utterance against a live environment."""

from __future__ import annotations

import json

from aieng.syn_data.synbench.agents.prompts import agent_system_prompt
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.llm.client import LLMClient, get_client
from aieng.syn_data.synbench.llm.tools import tool_specs_to_openai
from aieng.syn_data.synbench.schemas.domain import DomainBundle
from aieng.syn_data.synbench.schemas.tasks import Task


class ToolCallingLoop:
    """
    Multi-turn tool-calling loop with a live environment.

        Uses: llm.client, llm.tools, agents.prompts, agents.session, environment.core
        Used by: SingleToolAgent, AgentPipeline (as executor)
    """

    def __init__(
        self,
        domain: DomainBundle,
        client: LLMClient | None = None,
        max_turns: int = 12,
    ):
        self.domain = domain
        self.client = client or get_client()
        self.max_turns = max_turns

    def run(
        self,
        task: Task,
        session: AgentSession | None = None,
        *,
        plan: str = "",
        initial_user_message: str | None = None,
    ) -> AgentSession:
        """Run tool-calling until a final text reply for the current user turn.

        ``initial_user_message`` semantics:
        - ``None``: append ``task.user_scenario.initial_message`` if missing
          (standalone loop usage).
        - ``""``: do not append a user message (dialogue helper already did).
        - non-empty str: append that content if not already present.
        """
        session = session or AgentSession(domain=self.domain, task=task)
        env = session.ensure_env()
        tools = tool_specs_to_openai(self.domain.tools)

        # Keep / refresh the system prompt at the front of the transcript.
        system = agent_system_prompt(self.domain, task, extra=plan)
        if session.messages and session.messages[0].get("role") == "system":
            session.messages[0] = {"role": "system", "content": system}
        else:
            session.messages.insert(0, {"role": "system", "content": system})

        # None → seed from task; "" → skip (outer dialogue already appended).
        if initial_user_message is None:
            user_msg = task.user_scenario.initial_message
        else:
            user_msg = initial_user_message
        if user_msg and not any(
            m.get("role") == "user" and m.get("content") == user_msg
            for m in session.messages
        ):
            session.messages.append({"role": "user", "content": user_msg})

        for _ in range(self.max_turns):
            # Tool calls → execute and feed results back; a text reply ends
            # this user turn. Outer dialogue helpers may then append further
            # user messages on the same session.
            response = self.client.complete(session.messages, tools=tools)

            if response.tool_calls:
                raw = response.raw_message or {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": a.name,
                                "arguments": json.dumps(a.arguments),
                            },
                        }
                        for i, a in enumerate(response.tool_calls)
                    ],
                }
                session.messages.append(raw)

                # Live dispatch: tool results feed back into the chat so the
                # LLM can plan the next turn. Scoring later replays
                # session.agent_actions on a fresh env — do not confuse the two.
                for i, action in enumerate(response.tool_calls):
                    session.agent_actions.append(action)
                    try:
                        # returns the result of the tool call which can be an error
                        # or the result (output) of the tool call which could be
                        # anything such as dictionary or list
                        result = env.dispatch(action)
                        result_str = json.dumps(result, default=str)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                    call_id = "call_0"
                    if raw.get("tool_calls") and i < len(raw["tool_calls"]):
                        call_id = raw["tool_calls"][i].get("id", f"call_{i}")
                    session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_str,
                        }
                    )
                continue

            if response.content:
                session.agent_messages.append(response.content)
                session.messages.append(
                    {"role": "assistant", "content": response.content}
                )
            break

        return session
