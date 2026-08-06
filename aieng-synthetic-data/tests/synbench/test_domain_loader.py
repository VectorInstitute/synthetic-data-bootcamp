"""Tests for loading and validating domain bundles."""

import pytest

from aieng.syn_data.synbench.domain.loader import (
    DomainLoadError,
    load_domain,
    validate_domain,
)


def test_load_mock_retail(mock_retail_path):
    """The bundled domain loads with its tools, seeds, policy, and config."""
    bundle = load_domain(mock_retail_path)
    assert bundle.manifest.name == "mock_retail"
    assert len(bundle.tools) >= 4
    assert len(bundle.seed_tasks) == 4
    assert len(bundle.policy) > 0
    assert bundle.generation.primary_collection == "orders"


def test_validate_mock_retail(mock_retail_path):
    """The bundled domain passes validation with no errors."""
    assert validate_domain(mock_retail_path) == []


def test_missing_file_raises(tmp_path):
    """A domain folder missing required files fails to load."""
    (tmp_path / "policy.md").write_text("x")
    with pytest.raises(DomainLoadError, match="missing required"):
        load_domain(tmp_path)
