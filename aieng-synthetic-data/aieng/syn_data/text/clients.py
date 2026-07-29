"""LLM client setup for teacher and judge models."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests
from json_repair import repair_json


logger = logging.getLogger(__name__)
# TODO: Change to INFO level for production
logger.setLevel(logging.DEBUG)

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
                f"Missing API key. Set {api_key_var} in your environment or in "
                "implementations/qa_text_generation/.env (see .env.example)."
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

        max_attempts = 5
        retryable_statuses = {429, 500, 502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
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
                choice = payload["choices"][0]
                content = choice["message"].get("content")
                if content is None:
                    finish_reason = choice.get("finish_reason")
                    has_tool_calls = bool(choice["message"].get("tool_calls"))
                    msg = (
                        f"Model {self.settings.model} returned null content "
                        f"(finish_reason={finish_reason!r}, tool_calls={has_tool_calls})."
                    )
                    raise ValueError(msg)
                return str(content)
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                status = (
                    exc.response.status_code
                    if isinstance(exc, requests.HTTPError) and exc.response is not None
                    else None
                )
                is_retryable = (
                    isinstance(exc, (requests.ConnectionError, requests.Timeout))
                    or status in retryable_statuses
                )
                if not is_retryable or attempt == max_attempts:
                    raise
                wait = min(2**attempt, 60)
                logger.warning(
                    "LLM request failed (attempt %d/%d, status=%s): %s. Retrying in %ds.",
                    attempt,
                    max_attempts,
                    status,
                    exc,
                    wait,
                )
                time.sleep(wait)

        msg = f"LLM request to {self.settings.model} exhausted retries."
        raise RuntimeError(msg)

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
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
        logger.debug(
            "*********** Extracted JSON payload: *********** \n%s\n*********** End of JSON payload ***********",
            cleaned,
        )
        try:
            parsed = json.loads(repair_json(cleaned))
        except json.JSONDecodeError as exc:
            snippet = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
            msg = f"Failed to parse model JSON ({exc.msg}): {snippet!r}"
            raise ValueError(msg) from exc
        if not isinstance(parsed, dict):
            raw_snippet = (raw[:300] + ("..." if len(raw) > 300 else "")) if raw else ""
            msg = (
                f"Model response did not contain a JSON object "
                f"(got {type(parsed).__name__}): {raw_snippet!r}"
            )
            raise ValueError(msg)
        return cast(dict[str, Any], parsed)


def create_teacher_client(
    *,
    model_var: str = "TEACHER_MODEL",
    default_model: str = "gpt-4o-mini",
) -> OpenAICompatibleClient:
    """
    Create the strong teacher model client for synthetic data generation.

    This function initializes and returns an OpenAI-compatible LLM client intended to act as the
    "teacher" during synthetic data generation. The specific model to use can be configured through
    an environment variable or a default is provided.

    Parameters
    ----------
    model_var : str, optional
        Name of the environment variable holding the desired model name. Defaults to "TEACHER_MODEL".
    default_model : str, optional
        Model to use if the environment variable is not set. Defaults to "gpt-4o-mini".

    Returns
    -------
    OpenAICompatibleClient
        An LLM client configured for use as the teacher in synthetic data workflows.

    Notes
    -----
    The returned client encapsulates credentials and endpoint information inferred from
    environment variables as per the LLMSettings class.
    """
    settings = LLMSettings.from_env(model_var=model_var, default_model=default_model)
    return OpenAICompatibleClient(settings)


def create_judge_client(
    *,
    model_var: str = "JUDGE_MODEL",
    default_model: str = "gpt-4o-mini",
) -> OpenAICompatibleClient:
    """
    Create the LLM client for use as a judge model in evaluation tasks.

    This function initializes and returns an OpenAI-compatible LLM client that will be
    used in the "judge" role for synthetic data evaluation workflows. The selected model
    can be configured through an environment variable or a default model is used if none
    is specified.

    Parameters
    ----------
    model_var : str, optional
        Name of the environment variable that contains the judge model name to use.
        Defaults to "JUDGE_MODEL".
    default_model : str, optional
        Model to use as fallback if the environment variable is not set.
        Defaults to "gpt-4o-mini".

    Returns
    -------
    OpenAICompatibleClient
        An LLM client configured with credentials, endpoint, and model specification, ready
        for use in LLM-based judge evaluation flows.

    Notes
    -----
    The returned client infers model and environment configuration from the specified
    environment variable, using the `LLMSettings` class for settings discovery. This facilitates
    flexible and reproducible judge model setup in varying compute or deployment contexts.
    """
    settings = LLMSettings.from_env(model_var=model_var, default_model=default_model)
    return OpenAICompatibleClient(settings)
