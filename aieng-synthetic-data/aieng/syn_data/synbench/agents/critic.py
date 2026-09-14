"""Critic role: reviews an agent draft reply and approves or requests revisions."""

from __future__ import annotations

from aieng.syn_data.synbench.agents.prompts import critic_system_prompt
from aieng.syn_data.synbench.agents.session import AgentSession
from aieng.syn_data.synbench.llm.client import LLMClient, get_client


class Critic:
    """LLM reviewer that approves or rejects the executor's draft reply."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or get_client()

    def review(self, session: AgentSession) -> tuple[bool, str]:
        """Review the session's plan, tool trace, and draft reply.

        Returns
        -------
        tuple[bool, str]
            Whether the draft was approved, and the critic's raw notes.
        """
        trace = "\n".join(f"- {a.name}({a.arguments})" for a in session.agent_actions)
        draft = session.agent_messages[-1] if session.agent_messages else ""
        user_content = "\n".join(
            [
                f"Plan:\n{session.plan}",
                f"Tool trace:\n{trace or '(none)'}",
                f"Draft reply:\n{draft}",
            ]
        )
        messages = [
            {
                "role": "system",
                "content": critic_system_prompt(session.domain, session.task),
            },
            {"role": "user", "content": user_content},
        ]
        response = self.client.complete(messages)
        text = (response.content or "").strip()
        approved = text.upper().startswith("APPROVE")
        return approved, text
