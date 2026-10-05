"""Logging setup and retry helper shared by the pipeline scripts."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from edgar.exceptions import NotFoundError

from src.config import LOG_DIR


T = TypeVar("T")


def setup_logging(name: str) -> logging.Logger:
    """Log to the console and to logs/<name>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(console)
        logger.addHandler(file_handler)
    return logger


def with_retries(
    func: Callable[[], T],
    description: str,
    logger: logging.Logger,
    attempts: int = 4,
    base_delay: float = 2.0,
) -> T:
    """Call func, retrying failures with exponential backoff (2s, 4s, 8s by default).

    edgartools already retries network errors and HTTP 429 responses. This adds a
    second layer for other failures. NotFoundError is raised immediately because
    retrying will not help.
    """
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except NotFoundError:
            raise
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.0fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")
