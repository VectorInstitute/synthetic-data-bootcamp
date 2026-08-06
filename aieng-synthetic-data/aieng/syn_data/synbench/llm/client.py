"""Client protocol and factory shared by the real and mock LLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aieng.syn_data.synbench.llm.config import get_model, is_mock_llm
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
    """Build the mock or remote client according to the environment."""
    # Imported lazily: mock_client imports this module, and chat_client pulls in
    # the optional openai dependency.
    if is_mock_llm():
        from aieng.syn_data.synbench.llm.mock_client import (  # noqa: PLC0415
            MockLLMClient,
        )

        return MockLLMClient()

    from aieng.syn_data.synbench.llm.chat_client import ChatClient  # noqa: PLC0415

    return ChatClient(model=model or get_model())
