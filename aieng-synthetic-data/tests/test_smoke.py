"""Smoke tests for aieng-synthetic-data package."""


def test_syn_data_subpackages_importable() -> None:
    """Package submodules should be importable."""
    import aieng.syn_data.image  # noqa: F401
    import aieng.syn_data.text  # noqa: F401
