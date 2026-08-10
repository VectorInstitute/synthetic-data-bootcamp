"""Build YOLO detection datasets from Notebook 2 manifests and run short fine-tunes."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from edgecase_synthesis.eda import write_json


def canonicalize_label(label: str) -> str:
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class DatasetBuildStats:
    n_images: int = 0
    n_boxes: int = 0
    n_skipped_boxes: int = 0
    n_empty_label_images: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    n_real: int = 0
    n_synthetic: int = 0


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


def build_yolo_dataset(
    *,
    train_manifest: list[dict[str, Any]],
    test_manifest: list[dict[str, Any]],
    out_dir: Path | str,
    class_names: list[str],
    aliases: dict[str, list[str]] | None = None,
    include_synthetic: bool = True,
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

    def _row_has_target_box(row: dict[str, Any]) -> bool:
        for box in row.get("boxes") or []:
            if box_to_class_id(str(box.get("label", "")), lookup) is not None:
                return True
        return False

    def _ingest(rows: list[dict[str, Any]], split: str, stats: DatasetBuildStats) -> None:
        for row in rows:
            kind = str(row.get("split", "real")).lower()
            if split == "train" and kind == "synthetic" and not include_synthetic:
                continue
            if split == "val" and kind == "synthetic":
                # Never evaluate on synthetic / auto-labels.
                continue
            if (
                split == "train"
                and kind == "synthetic"
                and drop_empty_synthetic
                and not _row_has_target_box(row)
            ):
                stats.n_skipped_boxes += 1  # reuse counter: empty synth dropped
                continue
            src = Path(str(row["path"]))
            if not src.exists():
                raise FileNotFoundError(f"Manifest image missing: {src}")
            stem = src.stem
            dst_img = root / "images" / split / src.name
            # Avoid collisions if two sources share a filename.
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
            else:
                stats.n_real += 1

    _ingest(train_manifest, "train", train_stats)
    _ingest(test_manifest, "val", val_stats)

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
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        root / "build_stats.json",
        {
            "dataset_name": dataset_name,
            "include_synthetic": include_synthetic,
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
    from ultralytics import YOLO

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
    from ultralytics import YOLO

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
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names)]
    else:
        class_names = list(names)

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


def plot_map_comparison(runs: list[TrainResult], *, title: str = "Detector comparison", ax=None):
    """Grouped bars: mAP50 + per-class AP50 for each run."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    class_keys: list[str] = []
    for run in runs:
        for cls in (run.metrics.get("per_class_ap50") or {}):
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
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    saved: list[Path] = []
    for path in image_paths[:max_images]:
        kwargs: dict[str, Any] = {"conf": conf, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = model.predict(str(path), **kwargs)
        if not results:
            continue
        plotted = results[0].plot()  # BGR ndarray
        dest = out_dir / f"pred_{path.stem}.jpg"
        Image.fromarray(plotted[:, :, ::-1]).save(dest, quality=92)
        saved.append(dest)
    return saved
