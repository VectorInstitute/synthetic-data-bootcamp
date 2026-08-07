"""Shared fixtures for synbench tests."""

from pathlib import Path

import pytest


# Repository root (synthetic-data-bootcamp/)
ROOT = Path(__file__).resolve().parents[3]
MOCK_RETAIL = (
    ROOT / "implementations" / "agent_benchmark_generation" / "domains" / "mock_retail"
)


@pytest.fixture(autouse=True)
def mock_llm_env(monkeypatch, request):
    """Force the mock LLM client for every non-integration test."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setenv("MOCK_LLM", "1")


@pytest.fixture
def mock_retail_path():
    """Path to the bundled ``mock_retail`` domain."""
    return MOCK_RETAIL


@pytest.fixture
def project_root():
    """Path to the repository root."""
    return ROOT
