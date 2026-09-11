"""Build YOLO detection datasets from Notebook 2 manifests and run short fine-tunes."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from ultralytics import YOLO  # type: ignore[attr-defined, unused-ignore]

from edgecase_synthesis.data.eda import write_json


def canonicalize_label(label: str) -> str:
    """Canonicalize a label for matching."""
    return str(label).lower().strip().replace("-", " ").replace("_", " ")


def build_alias_lookup(
    class_names: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """Map any accepted label string → class index.

    ``aliases`` maps canonical class name → list of synonyms (spaces/underscores ok).
    Each class name is always included as its own alias.
    """
    lookup: dict[str, int] = {}
    aliases = aliases or {}
    for idx, name in enumerate(class_names):
        synonyms = [name, *list(aliases.get(name, []) or [])]
        for syn in synonyms:
            lookup[canonicalize_label(syn)] = idx
    return lookup


def box_to_class_id(label: str, lookup: dict[str, int]) -> int | None:
    """Map a detection box to its class identifier."""
    return lookup.get(canonicalize_label(label))


def xyxy_to_yolo(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Convert pixel xyxy → YOLO normalized cx, cy, w, h. Returns None if invalid."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    nw = bw / width
    nh = bh / height
    # Clip to [0, 1] after normalize.
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    if nw <= 0.0 or nh <= 0.0:
        return None
    return cx, cy, nw, nh


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    """Load a dataset manifest."""
    return cast(list[dict[str, Any]], json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class DatasetBuildStats:
    """Represent DatasetBuildStats configuration and behavior."""

    n_images: int = 0
    n_boxes: int = 0
    n_skipped_boxes: int = 0
    n_empty_label_images: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    n_real: int = 0
    n_synthetic: int = 0
    n_scene: int = 0
    n_scene_dropped: int = 0


def _limit_scene_rows(
    rows: list[dict[str, Any]],
    max_scene_images: int | None,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int, int]:
    """Keep at most ``max_scene_images`` real ``tag=scene`` rows (background negatives).

    Synthetic and rare-tagged rows are always kept. Returns
    ``(filtered_rows, n_scene_kept, n_scene_dropped)``.
    """
    if max_scene_images is None:
        n_scene = sum(
            1
            for r in rows
            if str(r.get("split", "real")).lower() != "synthetic" and str(r.get("tag", "")).lower() == "scene"
        )
        return rows, n_scene, 0

    cap = max(0, int(max_scene_images))
    scenes: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("split", "real")).lower()
        tag = str(row.get("tag", "")).lower()
        if kind != "synthetic" and tag == "scene":
            scenes.append(row)
        else:
            rest.append(row)
    rng = random.Random(int(seed))
    rng.shuffle(scenes)
    kept = scenes[:cap]
    dropped = len(scenes) - len(kept)
    # Rare/synth first (signal), then capped backgrounds.
    return rest + kept, len(kept), dropped


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_label_file(
    path: Path,
    boxes: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    lookup: dict[str, int],
    class_names: list[str],
    stats: DatasetBuildStats,
) -> int:
    """Write one YOLO ``.txt`` label file. Returns number of kept boxes."""
    lines: list[str] = []
    for box in boxes:
        cid = box_to_class_id(str(box.get("label", "")), lookup)
        if cid is None:
            stats.n_skipped_boxes += 1
            continue
        yolo = xyxy_to_yolo(box["bbox_xyxy"], width, height)
        if yolo is None:
            stats.n_skipped_boxes += 1
            continue
        cx, cy, nw, nh = yolo
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        name = class_names[cid]
        stats.boxes_per_class[name] = stats.boxes_per_class.get(name, 0) + 1
        stats.n_boxes += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not lines:
        stats.n_empty_label_images += 1
    return len(lines)


def _synth_class_key(row: dict[str, Any]) -> str:
    return str(row.get("anomaly_id") or row.get("tag") or "unknown")


def count_usable_synthetic(
    train_manifest: list[dict[str, Any]],
    *,
    class_names: list[str],
    aliases: dict[str, list[str]] | None = None,
    drop_empty_synthetic: bool = True,
) -> dict[str, int]:
    """Count train-split synthetic rows per anomaly that would be kept for YOLO.

    Empty / non-target synth is excluded when ``drop_empty_synthetic`` is True so
    half/full caps match what Notebook 3 actually trains on.
    """
    lookup = build_alias_lookup(class_names, aliases)
    counts: dict[str, int] = {}
    for row in train_manifest:
        if str(row.get("split", "real")).lower() != "synthetic":
            continue
        if drop_empty_synthetic:
            has_box = any(
                box_to_class_id(str(box.get("label", "")), lookup) is not None for box in (row.get("boxes") or [])
            )
            if not has_box:
                continue
        key = _synth_class_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def synth_caps_for_fraction(
    train_manifest: list[dict[str, Any]],
    *,
    fraction: float,
    class_names: list[str],
    aliases: dict[str, list[str]] | None = None,
    drop_empty_synthetic: bool = True,
) -> dict[str, int]:
    """Calculate per-class synthetic caps for an ablation fraction."""
    if fraction < 0 or fraction > 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    available = count_usable_synthetic(
        train_manifest,
        class_names=class_names,
        aliases=aliases,
        drop_empty_synthetic=drop_empty_synthetic,
    )
    return {key: int(n * fraction) for key, n in available.items()}


def _resolve_synth_cap(
    key: str,
    caps: int | dict[str, int] | None,
) -> int | None:
    if caps is None:
        return None
    if isinstance(caps, dict):
        if key not in caps:
            return None  # class absent from fraction map → keep all of that key
        return int(caps[key])
    return int(caps)


def _should_skip_yolo_row(
    row: dict[str, Any],
    *,
    split: str,
    include_synthetic: bool,
    drop_empty_synthetic: bool,
    lookup: dict[str, int],
    resolved_caps: int | dict[str, int] | None,
    synth_kept: dict[str, int],
    stats: DatasetBuildStats,
) -> bool:
    """Apply synthetic filtering and per-class caps to one manifest row."""
    kind = str(row.get("split", "real")).lower()
    if split == "train" and kind == "synthetic" and not include_synthetic:
        return True
    if split == "val" and kind == "synthetic":
        return True
    if split == "train" and kind == "synthetic" and drop_empty_synthetic:
        has_target = any(
            box_to_class_id(str(box.get("label", "")), lookup) is not None for box in row.get("boxes") or []
        )
        if not has_target:
            stats.n_skipped_boxes += 1
            return True
    if split == "train" and kind == "synthetic" and resolved_caps is not None:
        synth_key = _synth_class_key(row)
        cap = _resolve_synth_cap(synth_key, resolved_caps)
        return cap is not None and synth_kept.get(synth_key, 0) >= cap
    return False


def _ingest_yolo_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    root: Path,
    stats: DatasetBuildStats,
    lookup: dict[str, int],
    class_names: list[str],
    include_synthetic: bool,
    drop_empty_synthetic: bool,
    resolved_caps: int | dict[str, int] | None,
    synth_kept: dict[str, int],
    copy_images: bool,
) -> None:
    """Materialize one YOLO split and update its build statistics."""
    for row in rows:
        if _should_skip_yolo_row(
            row,
            split=split,
            include_synthetic=include_synthetic,
            drop_empty_synthetic=drop_empty_synthetic,
            lookup=lookup,
            resolved_caps=resolved_caps,
            synth_kept=synth_kept,
            stats=stats,
        ):
            continue
        kind = str(row.get("split", "real")).lower()
        src = Path(str(row["path"]))
        if not src.exists():
            raise FileNotFoundError(f"Manifest image missing: {src}")
        stem = src.stem
        dst_img = root / "images" / split / src.name
        if dst_img.exists() or (root / "labels" / split / f"{stem}.txt").exists():
            stem = f"{kind}_{stem}"
            dst_img = root / "images" / split / f"{stem}{src.suffix}"
        _link_or_copy(src, dst_img, copy=copy_images)
        with Image.open(src) as im:
            width, height = im.size
        write_yolo_label_file(
            root / "labels" / split / f"{stem}.txt",
            list(row.get("boxes") or []),
            width=width,
            height=height,
            lookup=lookup,
            class_names=class_names,
            stats=stats,
        )
        stats.n_images += 1
        if kind == "synthetic":
            stats.n_synthetic += 1
            if split == "train":
                key = _synth_class_key(row)
                synth_kept[key] = synth_kept.get(key, 0) + 1
        else:
            stats.n_real += 1
            if str(row.get("tag", "")).lower() == "scene":
                stats.n_scene += 1


def build_yolo_dataset(
    *,
    train_manifest: list[dict[str, Any]],
    test_manifest: list[dict[str, Any]],
    out_dir: Path | str,
    class_names: list[str],
    aliases: dict[str, list[str]] | None = None,
    include_synthetic: bool = True,
    max_synthetic_per_class: int | dict[str, int] | None = None,
    synthetic_fraction: float | None = None,
    max_scene_images: int | None = None,
    scene_sample_seed: int = 42,
    copy_images: bool = False,
    dataset_name: str = "edgecase",
    drop_empty_synthetic: bool = True,
) -> tuple[Path, DatasetBuildStats, DatasetBuildStats]:
    """Materialize Ultralytics YOLO folders + ``data.yaml``.

    Layout::

        out_dir/
          images/{train,val}/...
          labels/{train,val}/...
          data.yaml

    Test (real-only) is written to the ``val`` split used for Ultralytics eval.
    When ``drop_empty_synthetic`` is True, synth rows with zero target-class boxes
    are skipped (they cannot teach the detector).

    Cap synth with either:

    - ``synthetic_fraction`` (0–1): per-class floor of usable NB2 accepts
      (preferred for 50% / 100% of whatever Notebook 2 produced).
    - ``max_synthetic_per_class``: fixed int, or per-class ``dict``.

    When ``max_scene_images`` is set, at most that many real ``tag=scene`` train
    rows are kept as background negatives (avoids ~500 empty vs ~50 rare imbalance).
    """
    root = Path(out_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    lookup = build_alias_lookup(class_names, aliases)
    train_stats = DatasetBuildStats()
    val_stats = DatasetBuildStats()
    synth_kept: dict[str, int] = {}
    train_manifest, _n_scene_kept, n_scene_dropped = _limit_scene_rows(
        train_manifest,
        max_scene_images,
        seed=scene_sample_seed,
    )
    train_stats.n_scene_dropped = n_scene_dropped

    available_synth = count_usable_synthetic(
        train_manifest,
        class_names=class_names,
        aliases=aliases,
        drop_empty_synthetic=drop_empty_synthetic,
    )
    resolved_caps: int | dict[str, int] | None = max_synthetic_per_class
    if synthetic_fraction is not None:
        resolved_caps = synth_caps_for_fraction(
            train_manifest,
            fraction=float(synthetic_fraction),
            class_names=class_names,
            aliases=aliases,
            drop_empty_synthetic=drop_empty_synthetic,
        )

    for rows, split, stats in (
        (train_manifest, "train", train_stats),
        (test_manifest, "val", val_stats),
    ):
        _ingest_yolo_rows(
            rows,
            split=split,
            stats=stats,
            root=root,
            lookup=lookup,
            class_names=class_names,
            include_synthetic=include_synthetic,
            drop_empty_synthetic=drop_empty_synthetic,
            resolved_caps=resolved_caps,
            synth_kept=synth_kept,
            copy_images=copy_images,
        )

    data_yaml = root / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                names_block,
                "",
            ],
        ),
        encoding="utf-8",
    )
    write_json(
        root / "build_stats.json",
        {
            "dataset_name": dataset_name,
            "include_synthetic": include_synthetic,
            "max_synthetic_per_class": max_synthetic_per_class,
            "synthetic_fraction": synthetic_fraction,
            "available_synthetic_per_class": available_synth,
            "resolved_synthetic_caps": resolved_caps,
            "kept_synthetic_per_class": synth_kept,
            "max_scene_images": max_scene_images,
            "drop_empty_synthetic": drop_empty_synthetic,
            "class_names": class_names,
            "aliases": aliases or {},
            "train": train_stats.__dict__,
            "val": val_stats.__dict__,
        },
    )
    return data_yaml, train_stats, val_stats


@dataclass
class TrainResult:
    """Represent TrainResult configuration and behavior."""

    name: str
    weights: Path
    metrics: dict[str, Any]
    data_yaml: Path
    runs_dir: Path


def _metric_float(metrics: Any, *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if hasattr(metrics, key):
            val = getattr(metrics, key)
            if val is not None:
                return float(val)
        if isinstance(metrics, dict) and key in metrics:
            return float(metrics[key])
    # Ultralytics results sometimes expose box.map etc.
    box = getattr(metrics, "box", None)
    if box is not None:
        for key in keys:
            if hasattr(box, key):
                return float(getattr(box, key))
    return default


def train_detector(
    data_yaml: Path | str,
    *,
    name: str,
    project_dir: Path | str,
    model_name: str = "yolov8n.pt",
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    seed: int = 42,
    workers: int = 4,
    patience: int = 20,
) -> TrainResult:
    """Fine-tune an Ultralytics YOLO detector; return best weights + val metrics."""
    pass

    data_yaml = Path(data_yaml)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "seed": int(seed),
        "workers": int(workers),
        "patience": int(patience),
        "verbose": True,
    }
    if device is not None:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    runs_dir = project_dir / name
    weights = runs_dir / "weights" / "best.pt"
    if not weights.exists():
        weights = runs_dir / "weights" / "last.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights found under {runs_dir / 'weights'}")

    metrics = evaluate_detector(weights, data_yaml, device=device, imgsz=imgsz)
    return TrainResult(
        name=name,
        weights=weights,
        metrics=metrics,
        data_yaml=data_yaml,
        runs_dir=runs_dir,
    )


def evaluate_detector(
    weights: Path | str,
    data_yaml: Path | str,
    *,
    device: str | int | None = None,
    imgsz: int = 640,
    split: str = "val",
) -> dict[str, Any]:
    """Run Ultralytics val and return a flat metrics dict (mAP + per-class AP)."""
    pass

    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": int(imgsz),
        "split": split,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    results = model.val(**kwargs)
    names = getattr(results, "names", None) or {}
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    per_class: dict[str, float] = {}
    box = getattr(results, "box", None)
    ap50 = getattr(box, "ap50", None) if box is not None else None
    if ap50 is not None:
        try:
            for i, ap in enumerate(list(ap50)):
                label = class_names[i] if i < len(class_names) else str(i)
                per_class[label] = float(ap)
        except TypeError:
            pass

    return {
        "map50": _metric_float(results, "map50", default=_metric_float(box, "map50") if box else float("nan")),
        "map50_95": _metric_float(
            results,
            "map",
            "map50-95",
            default=_metric_float(box, "map") if box else float("nan"),
        ),
        "precision": _metric_float(results, "mp", default=_metric_float(box, "mp") if box else float("nan")),
        "recall": _metric_float(results, "mr", default=_metric_float(box, "mr") if box else float("nan")),
        "per_class_ap50": per_class,
        "class_names": class_names,
    }


def metrics_table(runs: list[TrainResult]) -> list[dict[str, Any]]:
    """Rows suitable for printing / plotting."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "run": run.name,
            "map50": run.metrics.get("map50"),
            "map50_95": run.metrics.get("map50_95"),
            "precision": run.metrics.get("precision"),
            "recall": run.metrics.get("recall"),
            "weights": str(run.weights),
        }
        for cls, ap in (run.metrics.get("per_class_ap50") or {}).items():
            row[f"ap50_{cls}"] = ap
        rows.append(row)
    return rows


def plot_map_comparison(
    runs: list[TrainResult],
    *,
    title: str = "Detector comparison",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot grouped mAP50 and per-class AP50 bars for each run."""
    pass
    pass

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = cast(Figure, ax.figure)

    class_keys: list[str] = []
    for run in runs:
        for cls in run.metrics.get("per_class_ap50") or {}:
            if cls not in class_keys:
                class_keys.append(cls)
    metric_keys = ["map50", *[f"ap50::{c}" for c in class_keys]]
    labels = ["mAP50", *[f"AP50 {c}" for c in class_keys]]

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(runs), 1)
    for i, run in enumerate(runs):
        vals = []
        for key in metric_keys:
            if key == "map50":
                vals.append(float(run.metrics.get("map50") or 0.0))
            else:
                cls = key.split("::", 1)[1]
                vals.append(float((run.metrics.get("per_class_ap50") or {}).get(cls) or 0.0))
        ax.bar(x + i * width, vals, width, label=run.name)
    ax.set_xticks(x + width * (len(runs) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def predict_gallery(
    weights: Path | str,
    image_paths: list[Path],
    *,
    out_dir: Path | str,
    conf: float = 0.25,
    device: str | int | None = None,
    max_images: int = 8,
) -> list[Path]:
    """Save prediction overlays for a handful of test images."""
    pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    saved: list[Path] = []
    for path in image_paths[:max_images]:
        kwargs: dict[str, Any] = {"conf": conf, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = list(model.predict(str(path), **kwargs))
        if not results:
            continue
        result: Any = results[0]
        plotted = result.plot()  # BGR ndarray
        dest = out_dir / f"pred_{path.stem}.jpg"
        Image.fromarray(plotted[:, :, ::-1]).save(dest, quality=92)
        saved.append(dest)
    return saved
