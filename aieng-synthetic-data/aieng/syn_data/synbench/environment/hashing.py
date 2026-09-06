"""Deterministic hashing of environment database state."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, float) and obj == int(obj):
        return int(obj)
    return obj


def db_hash(db: dict[str, Any]) -> str:
    """Deterministic hash of DB state (order-independent keys)."""
    canonical = _canonicalize(db)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
