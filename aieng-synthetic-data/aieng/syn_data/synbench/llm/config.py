"""Environment-driven LLM configuration (model, endpoint, credentials)."""

from __future__ import annotations

import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-2.0-flash"

# Loads your environment variables from the .env file
load_dotenv()


def get_model() -> str:
    """Model name from ``SYNBENCH_MODEL``, falling back to ``DEFAULT_MODEL``."""
    return os.environ.get("SYNBENCH_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    """Chat completions endpoint from ``SYNBENCH_BASE_URL`` or ``BASE_URL``."""
    return os.environ.get(
        "SYNBENCH_BASE_URL",
        os.environ.get("BASE_URL", "https://proxy.vectorinstitute.ai/v1"),
    )


def get_api_key() -> str:
    """Read the API key from ``OPENAI_API_KEY`` or ``SYNBENCH_API_KEY``."""
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNBENCH_API_KEY", "")
