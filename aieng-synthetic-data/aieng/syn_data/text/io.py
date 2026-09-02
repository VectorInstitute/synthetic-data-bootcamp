"""JSONL and JSON file helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar, cast


T = TypeVar("T")


def ensure_parent(path: Path) -> Path:
    """Create parent directories for a file path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    """Load a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return cast(dict[str, Any] | list[Any], json.load(handle))


def write_json(
    path: Path, payload: dict[str, Any] | list[Any], *, indent: int = 2
) -> Path:
    """Write a JSON file, creating parent directories as needed."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    append: bool = False,
) -> Path:
    """Write records to a JSONL file."""
    ensure_parent(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def load_typed_jsonl(
    path: Path,
    factory: Callable[[dict[str, Any]], T],
) -> list[T]:
    """Load JSONL records into typed objects."""
    return [factory(record) for record in read_jsonl(path)]


def save_typed_jsonl(
    path: Path,
    records: Iterable[T],
    *,
    to_dict: Callable[[T], dict[str, Any]],
    append: bool = False,
) -> Path:
    """Save typed objects to JSONL."""
    return write_jsonl(path, (to_dict(record) for record in records), append=append)
