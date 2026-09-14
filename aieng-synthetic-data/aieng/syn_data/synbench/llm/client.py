"""Client protocol and factory for the OpenAI-compatible LLM backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aieng.syn_data.synbench.llm.config import get_model
from aieng.syn_data.synbench.schemas.actions import Action


@dataclass
class LLMResponse:
    """One model completion: text, requested tool calls, and the raw message."""

    content: str = ""
    tool_calls: list[Action] = field(default_factory=list)
    raw_message: Any = None


class LLMClient(Protocol):
    """Minimal interface every LLM backend must implement."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return the model's completion for ``messages``."""
        ...

    def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Complete in JSON mode and return the decoded object."""
        ...


def get_client(model: str | None = None) -> LLMClient:
    """Build the remote OpenAI-compatible chat client."""
    # Imported lazily so chat_client's optional openai dependency is not
    # required just to import this module (e.g. for LLMClient / LLMResponse).
    from aieng.syn_data.synbench.llm.chat_client import ChatClient  # noqa: PLC0415

    return ChatClient(model=model or get_model())
