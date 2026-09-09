"""Onboarding gate: verify bootcamp API keys against live services.

Run automatically on Coder workspace startup, or manually with
``onboard --bootcamp-name agentic-forecasting --test-script tests/test_integration.py``.

Every variable in ``.env.example`` is checked except ``FRED_API_KEY``.
"""

import os

import httpx
import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.integration_test


load_dotenv()

def test_openai_api_key() -> None:
    """Test that OPENAI_API_KEY is valid by making a minimal API call."""
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    response = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
        },
        timeout=30,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.is_error:
        pytest.fail(f"{response.status_code} from {response.url}: {response.text}")
    data = response.json()

    assert data.get("choices"), "OpenAI API returned no choices. Check your API key."