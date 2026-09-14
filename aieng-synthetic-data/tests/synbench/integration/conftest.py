"""Fixtures that point integration tests at a real LLM provider."""

import pytest

from aieng.syn_data.synbench.llm.config import get_api_key


@pytest.fixture(autouse=True)
def require_api_key():
    """Skip the test when no API key is configured."""
    if not get_api_key():
        pytest.skip("OPENAI_API_KEY or SYNBENCH_API_KEY required for integration tests")
