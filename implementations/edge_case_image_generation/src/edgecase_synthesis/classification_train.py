"""Build image-classification datasets from Notebook 2 manifests and run short fine-tunes."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgecase_synthesis.detection_train import TrainResult, _link_or_copy, load_manifest
from edgecase_synthesis.eda import write_json

__all__ = [
    "ClsDatasetBuildStats",
    "build_cls_dataset",
    "train_classifier",
    "evaluate_classifier",
    "metrics_table",
    "plot_metrics_comparison",
    "predict_gallery",
    "load_manifest",
]


@dataclass
class ClsDatasetBuildStats:
    n_images: int = 0
    n_skipped: int = 0
    images_per_class: dict[str, int] = field(default_factory=dict)
    n_real: int = 0
    n_synthetic: int = 0


def _row_label(row: dict[str, Any], class_names: list[str]) -> str | None:
    tag = str(row.get("tag") or row.get("anomaly_id") or "").strip()
    if tag in class_names:
        return tag
    return None


def build_cls_dataset(
    *,
    train_manifest: list[dict[str, Any]],
    test_manifest: list[dict[str, Any]],
    out_dir: Path | str,
    class_names: list[str],
    include_synthetic: bool = True,
    copy_images: bool = False,
    dataset_name: str = "edgecase_cls",
) -> tuple[Path, ClsDatasetBuildStats, ClsDatasetBuildStats]:
    """Materialize Ultralytics classification folders.

    Layout::

        out_dir/
          train/<class_name>/...
          val/<class_name>/...
          labels.json

    Image label = manifest ``tag`` (``scene``, ``traffic_cone``, ``ground_animal``, …).
    Test (real-only) is written to ``val``.
    """
    root = Path(out_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        for cls in class_names:
            (root / split / cls).mkdir(parents=True, exist_ok=True)

    train_stats = ClsDatasetBuildStats()
    val_stats = ClsDatasetBuildStats()
    label_records: dict[str, list[dict[str, str]]] = {"train": [], "val": []}

    def _ingest(rows: list[dict[str, Any]], split: str, stats: ClsDatasetBuildStats) -> None:
        for row in rows:
            kind = str(row.get("split", "real")).lower()
            if split == "train" and kind == "synthetic" and not include_synthetic:
                continue
            if split == "val" and kind == "synthetic":
                continue
            label = _row_label(row, class_names)
            if label is None:
                stats.n_skipped += 1
                continue
            src = Path(str(row["path"]))
            if not src.exists():
                raise FileNotFoundError(f"Manifest image missing: {src}")
            stem = src.stem
            dst = root / split / label / src.name
            if dst.exists():
                stem = f"{kind}_{stem}"
                dst = root / split / label / f"{stem}{src.suffix}"
            _link_or_copy(src, dst, copy=copy_images)
            label_records[split].append({"path": str(dst), "label": label})
            stats.n_images += 1
            stats.images_per_class[label] = stats.images_per_class.get(label, 0) + 1
            if kind == "synthetic":
                stats.n_synthetic += 1
            else:
                stats.n_real += 1

    _ingest(train_manifest, "train", train_stats)
    _ingest(test_manifest, "val", val_stats)

    write_json(
        root / "build_stats.json",
        {
            "dataset_name": dataset_name,
            "include_synthetic": include_synthetic,
            "class_names": class_names,
            "train": train_stats.__dict__,
            "val": val_stats.__dict__,
        },
    )
    write_json(root / "labels.json", label_records)
    return root, train_stats, val_stats


def _metric_float(metrics: Any, *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if hasattr(metrics, key):
            val = getattr(metrics, key)
            if val is not None:
                return float(val)
        if isinstance(metrics, dict) and key in metrics:
            return float(metrics[key])
    return default


def _collect_val_predictions(
    weights: Path | str,
    dataset_root: Path | str,
    *,
    class_names: list[str],
    device: str | int | None = None,
    imgsz: int = 224,
) -> tuple[list[str], list[str]]:
    """Return parallel lists of true labels and predicted labels on ``val/``."""
    from ultralytics import YOLO

    dataset_root = Path(dataset_root)
    model = YOLO(str(weights))
    y_true: list[str] = []
    y_pred: list[str] = []
    for cls in class_names:
        folder = dataset_root / "val" / cls
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            kwargs: dict[str, Any] = {"verbose": False}
            if device is not None:
                kwargs["device"] = device
            results = model.predict(str(path), imgsz=int(imgsz), **kwargs)
            if not results:
                continue
            pred_idx = int(results[0].probs.top1)
            names = results[0].names or {}
            pred_label = names.get(pred_idx, class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx))
            y_true.append(cls)
            y_pred.append(str(pred_label))
    return y_true, y_pred


def _per_class_prf(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """One-vs-rest precision / recall / F1 per class (no sklearn dependency)."""
    out: dict[str, dict[str, float]] = {}
    for cls in class_names:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred, strict=False):
            if p == cls and t == cls:
                tp += 1
            elif p == cls and t != cls:
                fp += 1
            elif p != cls and t == cls:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)}
    return out


def train_classifier(
    dataset_root: Path | str,
    *,
    name: str,
    project_dir: Path | str,
    class_names: list[str],
    model_name: str = "yolov8n-cls.pt",
    epochs: int = 40,
    imgsz: int = 224,
    batch: int = 16,
    device: str | int | None = None,
    seed: int = 42,
    workers: int = 4,
    patience: int = 15,
) -> TrainResult:
    """Fine-tune an Ultralytics YOLO classifier; return best weights + val metrics."""
    from ultralytics import YOLO

    dataset_root = Path(dataset_root)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    train_kwargs: dict[str, Any] = {
        "data": str(dataset_root),
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

    metrics = evaluate_classifier(
        weights,
        dataset_root,
        class_names=class_names,
        device=device,
        imgsz=imgsz,
    )
    return TrainResult(
        name=name,
        weights=weights,
        metrics=metrics,
        data_yaml=dataset_root,
        runs_dir=runs_dir,
    )


def evaluate_classifier(
    weights: Path | str,
    dataset_root: Path | str,
    *,
    class_names: list[str],
    device: str | int | None = None,
    imgsz: int = 224,
) -> dict[str, Any]:
    """Run Ultralytics cls val + per-class precision/recall/F1 on the val split."""
    from ultralytics import YOLO

    dataset_root = Path(dataset_root)
    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "data": str(dataset_root),
        "imgsz": int(imgsz),
        "split": "val",
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    results = model.val(**kwargs)

    y_true, y_pred = _collect_val_predictions(
        weights,
        dataset_root,
        class_names=class_names,
        device=device,
        imgsz=imgsz,
    )
    per_class = _per_class_prf(y_true, y_pred, class_names)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    accuracy = correct / len(y_true) if y_true else float("nan")

    # Macro averages over classes with at least one val example.
    supported = [c for c in class_names if per_class[c]["support"] > 0]
    macro_f1 = (
        sum(per_class[c]["f1"] for c in supported) / len(supported) if supported else float("nan")
    )
    macro_recall = (
        sum(per_class[c]["recall"] for c in supported) / len(supported) if supported else float("nan")
    )

    return {
        "top1_acc": _metric_float(results, "top1", "accuracy_top1", default=accuracy),
        "top5_acc": _metric_float(results, "top5", "accuracy_top5", default=float("nan")),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "class_names": class_names,
        "confusion": {
            "true": dict(Counter(y_true)),
            "pred": dict(Counter(y_pred)),
        },
    }


def metrics_table(runs: list[TrainResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "run": run.name,
            "top1_acc": run.metrics.get("top1_acc"),
            "macro_f1": run.metrics.get("macro_f1"),
            "macro_recall": run.metrics.get("macro_recall"),
            "weights": str(run.weights),
        }
        for cls, stats in (run.metrics.get("per_class") or {}).items():
            row[f"f1_{cls}"] = stats.get("f1")
            row[f"recall_{cls}"] = stats.get("recall")
        rows.append(row)
    return rows


def plot_metrics_comparison(
    runs: list[TrainResult],
    *,
    class_names: list[str],
    title: str = "Classifier comparison",
    ax=None,
):
    """Grouped bars: top-1 accuracy + per-class F1."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure

    metric_keys = ["top1_acc", *[f"f1::{c}" for c in class_names]]
    labels = ["Top-1 acc", *[f"F1 {c}" for c in class_names]]

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(runs), 1)
    for i, run in enumerate(runs):
        vals = []
        for key in metric_keys:
            if key == "top1_acc":
                vals.append(float(run.metrics.get("top1_acc") or run.metrics.get("accuracy") or 0.0))
            else:
                cls = key.split("::", 1)[1]
                vals.append(float((run.metrics.get("per_class") or {}).get(cls, {}).get("f1") or 0.0))
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
    class_names: list[str],
    out_dir: Path | str,
    device: str | int | None = None,
    max_images: int = 8,
) -> list[Path]:
    """Save images with predicted class + confidence overlaid as text in filename."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    saved: list[Path] = []
    for path in image_paths[:max_images]:
        kwargs: dict[str, Any] = {"verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = model.predict(str(path), **kwargs)
        if not results:
            continue
        r0 = results[0]
        pred_idx = int(r0.probs.top1)
        conf = float(r0.probs.top1conf)
        names = r0.names or {}
        pred_label = names.get(pred_idx, class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx))
        plotted = r0.plot()  # BGR ndarray
        from PIL import Image

        dest = out_dir / f"pred_{path.stem}_{pred_label}_{conf:.2f}.jpg"
        Image.fromarray(plotted[:, :, ::-1]).save(dest, quality=92)
        saved.append(dest)
    return saved
