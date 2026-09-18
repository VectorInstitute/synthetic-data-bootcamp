"""LLM client backed by an OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from aieng.syn_data.synbench.llm.client import LLMResponse
from aieng.syn_data.synbench.llm.config import get_api_key, get_base_url
from aieng.syn_data.synbench.llm.parsing import (
    assistant_message_to_dict,
    tool_calls_from_message,
)


class ChatClient:
    """Remote LLM client using the OpenAI-compatible chat completions API."""

    def __init__(self, model: str):
        """Initialize the chat client with the given model.
        Args:
            model: The model to use for the chat client. Can be either the agent model
                or the generator model.
                  
        """
        self.model = model 
        self._api = OpenAI(
            base_url=get_base_url(),
            api_key=get_api_key(),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send ``messages`` to the model and return the parsed response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._api.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        tool_actions = tool_calls_from_message(message)
        return LLMResponse(
            content=content,
            tool_calls=tool_actions,
            raw_message=assistant_message_to_dict(message),
        )

    def complete_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Complete in JSON mode and return the decoded object."""
        resp = self.complete(messages, json_mode=True)
        if not resp.content:
            return {}
        parsed: Any = json.loads(resp.content)
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        return dict(parsed)
