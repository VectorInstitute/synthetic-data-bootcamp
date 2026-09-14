"""Shared fixtures for synbench tests."""

from pathlib import Path

import pytest


# Repository root (synthetic-data-bootcamp/)
ROOT = Path(__file__).resolve().parents[3]
MOCK_RETAIL = (
    ROOT / "implementations" / "agent_benchmark_generation" / "domains" / "mock_retail"
)
MOCK_SAAS_BILLING = (
    ROOT
    / "implementations"
    / "agent_benchmark_generation"
    / "domains"
    / "mock_saas_billing"
)


@pytest.fixture
def mock_retail_path():
    """Path to the bundled ``mock_retail`` domain."""
    return MOCK_RETAIL


@pytest.fixture
def mock_saas_billing_path():
    """Path to the bundled ``mock_saas_billing`` domain."""
    return MOCK_SAAS_BILLING


@pytest.fixture
def project_root():
    """Path to the repository root."""
    return ROOT
