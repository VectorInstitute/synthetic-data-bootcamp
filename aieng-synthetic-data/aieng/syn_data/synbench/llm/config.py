"""Environment-driven LLM configuration (model, endpoint, credentials)."""

from __future__ import annotations

import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3-flash-preview"

# Models that only run in thinking mode: the proxy sends a zero thinking budget
# unless the request carries an explicit ``reasoning_effort``.
THINKING_ONLY_MODEL_PREFIXES = ("gemini-3-pro", "gemini-3.1-pro")
DEFAULT_REASONING_EFFORT = "medium"

# Loads your environment variables from the .env file
load_dotenv()


def get_agent_model() -> str:
    """Model name from ``AGENT_MODEL``, falling back to ``DEFAULT_MODEL``."""
    return os.environ.get("AGENT_MODEL", DEFAULT_MODEL)


def get_generator_model() -> str:
    """Model name from ``GENERATOR_MODEL``, falling back to ``DEFAULT_MODEL``."""
    return os.environ.get("GENERATOR_MODEL", DEFAULT_MODEL)

def get_user_simulator_model() -> str:
    """Model name from ``USER_SIMULATOR_MODEL``, falling back to ``DEFAULT_MODEL``."""
    return os.environ.get("USER_SIMULATOR_MODEL", DEFAULT_MODEL)


def get_reasoning_effort(model: str) -> str | None:
    """Reasoning effort to send for ``model``, or ``None`` when unsupported.

    Thinking-only models reject a request without ``reasoning_effort``, while
    models such as ``gpt-4o`` reject the parameter itself. Override the value
    with ``REASONING_EFFORT`` (``minimal``, ``medium`` or ``high``).
    """
    if not model.startswith(THINKING_ONLY_MODEL_PREFIXES):
        return None
    return os.environ.get("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)


def get_base_url() -> str:
    """Chat completions endpoint from ``SYNBENCH_BASE_URL`` or ``BASE_URL``."""
    return os.environ.get(
        "SYNBENCH_BASE_URL",
        os.environ.get("BASE_URL", "https://proxy.vectorinstitute.ai/v1"),
    )


def get_api_key() -> str:
    """Read the API key from ``OPENAI_API_KEY`` or ``SYNBENCH_API_KEY``."""
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNBENCH_API_KEY", "")
