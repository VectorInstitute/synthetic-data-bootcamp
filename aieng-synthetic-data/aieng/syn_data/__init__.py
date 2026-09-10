"""Synthetic data generation utilities and shared logging setup."""

import logging


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_PACKAGE_LOGGER_NAME = __name__  # "aieng.syn_data"


def _setup_package_logger() -> None:
    """Attach a formatted handler to the `aieng.syn_data` logger.

    Submodules using `logging.getLogger(__name__)` inherit this handler
    automatically, so they always log with the configured format without
    needing any extra setup at call sites.
    """
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    if getattr(pkg_logger, "_aieng_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.propagate = False
    pkg_logger._aieng_configured = True  # type: ignore[attr-defined]


def configure_logging() -> None:
    """Configure the root logger as well (e.g. for app entry points)."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)


_setup_package_logger()
