"""Mid-run checkpoint / resume for Notebook 2 batch synthesis.

Writes under ``<nb2_dir>/checkpoint/`` so a killed job can skip finished
(anomaly, seed) pairs and rebuild ``AcceptedSample`` rows without re-export.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

from edgecase_synthesis.batch.export import AcceptedSample, ClassRunStats


def checkpoint_dir(nb2_dir: Path | str) -> Path:
    """Return the checkpoint directory."""
    path = Path(nb2_dir) / "checkpoint"
    path.mkdir(parents=True, exist_ok=True)
    return path


def accepted_path(nb2_dir: Path | str) -> Path:
    """Return the accepted-record path."""
    return checkpoint_dir(nb2_dir) / "accepted.jsonl"


def rejected_path(nb2_dir: Path | str) -> Path:
    """Return the rejected-record path."""
    return checkpoint_dir(nb2_dir) / "rejected.jsonl"


def state_path(nb2_dir: Path | str) -> Path:
    """Return the checkpoint state path."""
    return checkpoint_dir(nb2_dir) / "state.json"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a record to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_key(anomaly_id: str, source_stem: str) -> str:
    """Sample key."""
    return f"{anomaly_id}::{source_stem}"


def accepted_sample_to_row(sample: AcceptedSample) -> dict[str, Any]:
    """Serialize an accepted sample."""
    return asdict(sample)


def row_to_accepted_sample(row: dict[str, Any]) -> AcceptedSample:
    """Deserialize an accepted sample."""
    return AcceptedSample(
        sample_id=str(row.get("sample_id") or Path(str(row.get("image_name", ""))).stem),
        anomaly_id=str(row["anomaly_id"]),
        method=str(row.get("method") or "instruct"),
        source_stem=str(row["source_stem"]),
        image_name=str(row["image_name"]),
        boxes=list(row.get("boxes") or []),
        prompt=str(row.get("prompt") or ""),
        seed=int(row.get("seed") or 0),
        judge=dict(row.get("judge") or {}),
        variation=dict(row.get("variation") or {}),
        variation_index=row.get("variation_index"),
    )


def load_checkpoint(nb2_dir: Path | str) -> dict[str, Any]:
    """Load prior accepts/rejects/state for resume."""
    accepted_rows = read_jsonl(accepted_path(nb2_dir))
    rejected_rows = read_jsonl(rejected_path(nb2_dir))
    state: dict[str, Any] = {}
    sp = state_path(nb2_dir)
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
    accepted = [row_to_accepted_sample(r) for r in accepted_rows]
    accepted_keys = {sample_key(s.anomaly_id, s.source_stem) for s in accepted}
    rejected_keys = {
        sample_key(str(r["anomaly_id"]), str(r["source_stem"]))
        for r in rejected_rows
        if r.get("anomaly_id") and r.get("source_stem")
    }
    return {
        "accepted": accepted,
        "accepted_keys": accepted_keys,
        "rejected": rejected_rows,
        "rejected_keys": rejected_keys,
        "state": state,
    }


def save_state(nb2_dir: Path | str, payload: dict[str, Any]) -> Path:
    """Save state."""
    path = state_path(nb2_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_split_snapshot(
    nb2_dir: Path | str,
    *,
    train_real: dict[str, list[Path]],
    test: dict[str, list[Path]],
    seeds_by_anomaly: dict[str, list[Path]],
    method_map: dict[str, str],
    target_accepts: dict[str, int],
    config_snapshot: dict[str, Any],
) -> Path:
    """Persist path lists so CLI resume does not depend on notebook RAM."""

    def _paths(groups: dict[str, list[Path]]) -> dict[str, list[str]]:
        return {k: [str(p) for p in v] for k, v in groups.items()}

    payload = {
        "train_real": _paths(train_real),
        "test": _paths(test),
        "seeds_by_anomaly": _paths(seeds_by_anomaly),
        "method_map": dict(method_map),
        "target_accepts": {k: int(v) for k, v in target_accepts.items()},
        "config": config_snapshot,
    }
    path = checkpoint_dir(nb2_dir) / "split_snapshot.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_split_snapshot(nb2_dir: Path | str) -> dict[str, Any] | None:
    """Load a saved dataset split."""
    path = checkpoint_dir(nb2_dir) / "split_snapshot.json"
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def rebuild_stats(
    anomaly_ids: Iterable[str],
    *,
    accepted: list[AcceptedSample],
    rejected: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> dict[str, ClassRunStats]:
    """Rebuild run statistics from checkpoint records."""
    stats = {aid: ClassRunStats(anomaly_id=aid) for aid in anomaly_ids}
    for sample in accepted:
        st = stats.setdefault(sample.anomaly_id, ClassRunStats(anomaly_id=sample.anomaly_id))
        st.accepts += 1
        st.attempts += 1
    for row in rejected:
        aid = str(row.get("anomaly_id") or "")
        if not aid:
            continue
        st = stats.setdefault(aid, ClassRunStats(anomaly_id=aid))
        st.rejects += 1
        st.attempts += 1
    # Prefer persisted attempt counters when present (includes retries).
    saved = (state or {}).get("stats") or {}
    for aid, payload in saved.items():
        st = stats.setdefault(str(aid), ClassRunStats(anomaly_id=str(aid)))
        st.attempts = max(st.attempts, int(payload.get("attempts") or 0))
        st.accepts = max(st.accepts, int(payload.get("accepts") or 0))
        st.rejects = max(st.rejects, int(payload.get("rejects") or 0))
        st.retries = max(st.retries, int(payload.get("retries") or 0))
    return stats
