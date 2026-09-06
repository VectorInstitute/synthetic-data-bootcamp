"""Planner role: turns a customer message into a short numbered plan."""

from __future__ import annotations

from aieng.syn_data.synbench.agents.prompts import planner_system_prompt
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.llm.client import LLMClient, get_client


class Planner:
    """LLM planner that drafts the steps the executor should follow."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or get_client()

    def plan(self, session: AgentSession, user_message: str) -> str:
        """Return a plain-text plan for handling ``user_message``."""
        messages = [
            {
                "role": "system",
                "content": planner_system_prompt(session.domain, session.task),
            }
        ]
        # First append the hostory
        for m in session.messages:
            if m.get("role") == "assistant" and m.get("content"):
                messages.append({"role": "assistant", "content": m["content"]})
        # Then append the user message
        messages.append({"role": "user", "content": user_message})
        response = self.client.complete(messages)
        return (response.content or "").strip()
