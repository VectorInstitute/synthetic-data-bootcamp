"""Smoke tests for aieng-synthetic-data package."""

import aieng.syn_data.text


def test_syn_data_subpackages_have_docstrings() -> None:
    """Package submodules should be importable and documented."""
    assert aieng.syn_data.text.__doc__ is not None
    assert "text" in aieng.syn_data.text.__doc__.lower()
