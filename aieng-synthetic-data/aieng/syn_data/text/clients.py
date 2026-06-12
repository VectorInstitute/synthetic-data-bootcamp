"""LLM client setup for teacher and judge models."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests


logger = logging.getLogger(__name__)

_FENCED_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_json_text(raw: str) -> str:
    """Pull a JSON object string out of prose or markdown-fenced model output."""
    text = raw.strip()
    match = _FENCED_JSON.search(text)
    if match:
        return match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


class LLMClient(Protocol):
    """Minimal interface for text generation used across notebooks."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...


@dataclass
class LLMSettings:
    """Connection settings for an OpenAI-compatible chat API."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 120

    @classmethod
    def from_env(
        cls,
        *,
        api_key_var: str = "OPENAI_API_KEY",
        model_var: str = "OPENAI_MODEL",
        base_url_var: str = "OPENAI_BASE_URL",
        default_model: str = "gpt-4o-mini",
    ) -> LLMSettings:
        """Build settings from environment variables."""
        api_key = os.getenv(api_key_var, "")
        if not api_key:
            msg = (
                f"Missing API key. Set {api_key_var} in your environment or .env file."
            )
            raise ValueError(msg)
        return cls(
            api_key=api_key,
            model=os.getenv(model_var, default_model),
            base_url=os.getenv(base_url_var, "https://api.openai.com/v1").rstrip("/"),
        )


class OpenAICompatibleClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Call the chat completions API and return assistant text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        response = requests.post(
            f"{self.settings.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Request a JSON object response and parse it."""
        system_prompt = system or "You are a helpful assistant."
        system_prompt = (
            f"{system_prompt}\n"
            "Respond with a single JSON object only. "
            "No markdown fences, explanations, or extra text."
        )
        try:
            raw = self.complete(
                prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {400, 422}:
                raise
            logger.warning(
                "response_format=json_object not supported by %s; retrying without it.",
                self.settings.model,
            )
            raw = self.complete(
                prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        cleaned = extract_json_text(raw)
        logger.debug("Extracted JSON payload: %s", cleaned)
        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError as exc:
            snippet = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
            msg = f"Failed to parse model JSON ({exc.msg}): {snippet!r}"
            raise ValueError(msg) from exc


def create_teacher_client(
    *,
    model_var: str = "TEACHER_MODEL",
    default_model: str = "gpt-4o-mini",
) -> OpenAICompatibleClient:
    """Create the strong model used for synthetic data generation."""
    settings = LLMSettings.from_env(model_var=model_var, default_model=default_model)
    return OpenAICompatibleClient(settings)


def create_judge_client(
    *,
    model_var: str = "JUDGE_MODEL",
    default_model: str = "gpt-4o-mini",
) -> OpenAICompatibleClient:
    """Create the model used for LLM-as-judge evaluation."""
    settings = LLMSettings.from_env(model_var=model_var, default_model=default_model)
    return OpenAICompatibleClient(settings)
