"""Class-distribution EDA and stratified train/test splits (parameterized)."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from edgecase_synthesis.data import DetectionBox, IMAGE_EXTENSIONS, load_detection_labels


def stem_tag(path: Path | str, prefixes: list[str] | None = None) -> str:
    """Tag from filename stem using longest matching prefix.

    Accepts either bare tags (``traffic_cone``) or extract-style prefixes
    with a trailing underscore (``traffic_cone_`` from ``data.stem_prefixes``).
    ``traffic_cone_xyz`` → ``traffic_cone``.
    """
    stem = Path(path).stem
    if prefixes:
        normalized: list[tuple[str, str]] = []
        for raw in prefixes:
            p = str(raw)
            tag = p[:-1] if p.endswith("_") else p
            normalized.append((tag, p if p.endswith("_") else f"{p}_"))
        for tag, with_us in sorted(normalized, key=lambda t: len(t[0]), reverse=True):
            if stem == tag or stem.startswith(with_us):
                return tag
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[0]


def list_tagged_images(samples_dir: Path | str) -> list[Path]:
    root = Path(samples_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def group_by_tag(
    paths: Iterable[Path],
    *,
    tags: list[str] | None = None,
    prefixes: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Group image paths by stem tag. If ``tags`` is set, only those keys (empty lists ok).

    Pass ``prefixes`` (e.g. ``cfg.data.stem_prefixes``) so multi-word tags like
    ``traffic_cone`` resolve correctly.
    """
    match_prefixes = list(prefixes) if prefixes is not None else (list(tags) if tags else None)
    out: dict[str, list[Path]] = {t: [] for t in (tags or [])}
    for path in paths:
        tag = stem_tag(path, match_prefixes)
        if tags is not None and tag not in out:
            continue
        out.setdefault(tag, []).append(path)
    for key in out:
        out[key] = sorted(out[key])
    return out


def image_has_label(
    path: Path,
    labels: dict[str, list[DetectionBox]],
    class_names: list[str],
) -> bool:
    """True if labels.json entry for this file contains any of ``class_names``."""
    boxes = labels.get(path.name) or labels.get(path.stem) or []
    wanted = {c.lower() for c in class_names}
    return any(str(b.label).lower() in wanted for b in boxes)


def class_image_counts(
    paths_by_tag: dict[str, list[Path]],
    *,
    labels: dict[str, list[DetectionBox]] | None = None,
    rare_classes: list[str] | None = None,
) -> dict[str, int]:
    """Count images per tag bucket (filename prefix).

    Also adds ``clean_scene`` = scene-tagged images with none of ``rare_classes`` in labels
    when labels + rare_classes are provided.
    """
    counts = {tag: len(paths) for tag, paths in paths_by_tag.items()}
    if labels is not None and rare_classes and "scene" in paths_by_tag:
        clean = [
            p
            for p in paths_by_tag["scene"]
            if not image_has_label(p, labels, rare_classes)
        ]
        counts["clean_scene"] = len(clean)
    return counts


def counts_to_shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def plot_class_bars(
    counts: dict[str, int],
    *,
    title: str = "Class distribution (images)",
    ax=None,
):
    """Bar chart of image counts. Returns (fig, ax)."""
    import matplotlib.pyplot as plt

    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    bars = ax.bar(keys, vals, color="#3d7ea6")
    ax.set_ylabel("images")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return fig, ax


def stratified_holdout(
    paths_by_tag: dict[str, list[Path]],
    test_counts: dict[str, int],
    *,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Draw a stratified test set; remainder is train.

    ``test_counts`` maps tag → number of images to put in test.
    Raises if a tag does not have enough images.
    """
    rng = random.Random(int(seed))
    test: dict[str, list[Path]] = {}
    train: dict[str, list[Path]] = {}
    for tag, n_test in test_counts.items():
        pool = list(paths_by_tag.get(tag) or [])
        if len(pool) < n_test:
            raise ValueError(
                f"Need {n_test} images for test tag={tag!r}, found {len(pool)}. "
                "Re-run extract with higher caps or lower TEST counts."
            )
        shuffled = pool[:]
        rng.shuffle(shuffled)
        test[tag] = sorted(shuffled[:n_test])
        train[tag] = sorted(shuffled[n_test:])
    # Tags present in train pool but not requested for test stay fully in train.
    for tag, paths in paths_by_tag.items():
        if tag in test_counts:
            continue
        train[tag] = sorted(paths)
    return train, test


def flatten_tag_groups(groups: dict[str, list[Path]]) -> list[Path]:
    out: list[Path] = []
    for paths in groups.values():
        out.extend(paths)
    return sorted(out)


def pick_synth_seeds(
    scene_paths: list[Path],
    n_per_class: dict[str, int],
    *,
    labels: dict[str, list[DetectionBox]] | None = None,
    rare_classes: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Disjoint clean scene seeds per anomaly id.

    Prefers scenes without rare GT boxes when labels are provided.
    """
    rng = random.Random(int(seed))
    pool = list(scene_paths)
    if labels is not None and rare_classes:
        clean = [p for p in pool if not image_has_label(p, labels, rare_classes)]
        if sum(n_per_class.values()) <= len(clean):
            pool = clean
    rng.shuffle(pool)
    need = sum(int(v) for v in n_per_class.values())
    if len(pool) < need:
        raise ValueError(
            f"Need {need} scene seeds for synthesis, found {len(pool)}. "
            "Extract more scene_* images or lower N_SYNTH_PER_CLASS."
        )
    out: dict[str, list[Path]] = {}
    i = 0
    for anomaly_id, n in n_per_class.items():
        n = int(n)
        out[anomaly_id] = sorted(pool[i : i + n])
        i += n
    return out


def summarize_distribution(
    paths_by_tag: dict[str, list[Path]],
    *,
    focus_tags: list[str] | None = None,
) -> dict[str, Any]:
    if focus_tags is not None:
        counts = {t: len(paths_by_tag.get(t) or []) for t in focus_tags}
    else:
        counts = {t: len(p) for t, p in paths_by_tag.items()}
    shares = counts_to_shares(counts)
    return {
        "counts": counts,
        "shares": shares,
        "total": int(sum(counts.values())),
    }


def load_labels_for_dir(samples_dir: Path | str) -> dict[str, list[DetectionBox]]:
    return load_detection_labels(samples_dir)


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def clamp_counts(
    requested: dict[str, int],
    available: dict[str, int],
    *,
    min_remaining: int = 0,
) -> dict[str, int]:
    """Shrink requested counts so each key fits ``available[k] - min_remaining``."""
    out: dict[str, int] = {}
    for key, n in requested.items():
        have = int(available.get(key, 0))
        out[key] = max(0, min(int(n), max(0, have - min_remaining)))
    return out


def allocate_budget(requested: dict[str, int], budget: int) -> dict[str, int]:
    """Proportionally shrink non-negative requests so ``sum(out) <= budget``."""
    budget = max(0, int(budget))
    req = {k: max(0, int(v)) for k, v in requested.items()}
    total = sum(req.values())
    if total <= budget:
        return req
    if total == 0 or budget == 0:
        return {k: 0 for k in req}
    out = {k: int(budget * (v / total)) for k, v in req.items()}
    # Distribute leftover to keys that were truncated the most.
    while sum(out.values()) < budget:
        key = max(req.keys(), key=lambda k: req[k] - out[k])
        if out[key] >= req[key]:
            break
        out[key] += 1
    return out
