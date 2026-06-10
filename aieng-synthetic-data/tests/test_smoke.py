"""Smoke tests for aieng-synthetic-data package."""

import aieng.syn_data.image
import aieng.syn_data.text


def test_syn_data_subpackages_have_docstrings() -> None:
    """Package submodules should be importable and documented."""
    assert aieng.syn_data.image.__doc__ is not None
    assert aieng.syn_data.text.__doc__ is not None
    assert "image" in aieng.syn_data.image.__doc__.lower()
    assert "text" in aieng.syn_data.text.__doc__.lower()
