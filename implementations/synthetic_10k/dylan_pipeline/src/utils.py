"""Logging setup and retry helper shared by the pipeline scripts."""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from edgar.exceptions import NotFoundError

from src.config import LOG_DIR


T = TypeVar("T")


def load_dotenv(path: Path) -> None:
    """Minimal `KEY=VALUE`-per-line .env loader (no quoting/interpolation support --
    none of this project's secrets need it). Mirrors config.local.yaml's role for
    SEC identity: a gitignored local file for secrets that never touches git,
    consumed by RAG.chatbot / RAG.chatbot_server for DEEPSEEK_API_KEY /
    ANTHROPIC_API_KEY. An already-exported shell variable always wins.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def write_atomic(path: Path, content: str | bytes) -> None:
    """Write a file via a temp file so a crash never leaves a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(content, bytes):
        tmp.write_bytes(content)
    else:
        tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


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
