"""LLM-backed customer that drives multi-turn evaluation dialogues."""

from __future__ import annotations

import re
from typing import Any

from aieng.syn_data.synbench.agents.prompts import user_sim_system_prompt
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.llm.client import LLMClient, get_client


# Shown to the simulator when the executor produced no text reply, so the
# transcript never ends on the simulator's own turn.
AGENT_TURN_NUDGE = "(The agent is waiting for your reply.)"


class UserSimulator:
    """LLM-backed customer for multi-turn evaluation dialogues."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or get_client()

    def respond(self, session: AgentSession, last_agent_reply: str = "") -> str:
        """Produce the next customer utterance given the live session history.

        Does not re-emit ``initial_message``; that is seeded by the dialogue
        helper on turn 0. ``last_agent_reply`` is appended only when it is not
        already the last agent turn in ``session.messages``.
        """
        # System prompt includes user_name, personality_style, and clean instructions.
        messages = [
            {
                "role": "system",
                "content": user_sim_system_prompt(session.domain, session.task),
            },
        ]
        messages.extend(self._transcript(session, last_agent_reply))

        response = self.client.complete(messages)
        return self._clean(session, response.content or "")

    def is_done(self, message: str) -> bool:
        """Return whether ``message`` signals the end of the dialogue."""
        text = (message or "").strip()
        return not text or "[[DONE]]" in text

    def _transcript(
        self, session: AgentSession, last_agent_reply: str
    ) -> list[dict[str, Any]]:
        """Re-cast the agent-centric transcript from the customer's point of view.

        ``session.messages`` is written for the agent under test: the customer
        speaks as ``user`` and the agent replies as ``assistant``. The simulator
        *is* the customer, so the roles must be swapped before handing the
        history to its LLM — otherwise the model is asked to continue as the
        support agent and, because the list would end on its own turn, returns
        an empty completion.
        """
        turns: list[dict[str, Any]] = []
        for m in session.messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                turns.append({"role": "assistant", "content": content})
            elif role == "assistant" and not m.get("tool_calls"):
                # Assistant messages carrying tool_calls are internal executor
                # steps the customer never sees.
                turns.append({"role": "user", "content": content})

        reply = (last_agent_reply or "").strip()
        if reply and not (turns and turns[-1] == {"role": "user", "content": reply}):
            turns.append({"role": "user", "content": reply})

        turns = self._merge_consecutive(turns)
        if not turns or turns[-1]["role"] != "user":
            turns.append({"role": "user", "content": AGENT_TURN_NUDGE})
        return turns

    @staticmethod
    def _merge_consecutive(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse same-role runs so providers requiring alternation accept them."""
        merged: list[dict[str, Any]] = []
        for turn in turns:
            if merged and merged[-1]["role"] == turn["role"]:
                merged[-1]["content"] = f"{merged[-1]['content']}\n{turn['content']}"
            else:
                merged.append(dict(turn))
        return merged

    @staticmethod
    def _clean(session: AgentSession, content: str) -> str:
        """Drop a leading speaker prefix that models often prepend."""
        text = content.strip()
        names = ["Customer", "User"]
        user_name = session.task.user_scenario.user_name
        if user_name:
            names.append(user_name)
        for name in names:
            text = re.sub(
                rf"^\**{re.escape(name)}\**\s*:\s*", "", text, flags=re.IGNORECASE
            )
        return text.strip()
