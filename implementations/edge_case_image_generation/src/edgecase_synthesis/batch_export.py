"""Batch synthesis export helpers (manifests, label JSON, run stats)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from edgecase_synthesis.annotation import AnnotationResult, Detection
from edgecase_synthesis.data import DetectionBox
from edgecase_synthesis.eda import write_json
from edgecase_synthesis.generation import GenerationResult
from edgecase_synthesis.judge import JudgeResult


@dataclass
class ClassRunStats:
    anomaly_id: str
    attempts: int = 0
    accepts: int = 0
    rejects: int = 0
    retries: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / self.attempts if self.attempts else 0.0


@dataclass
class AcceptedSample:
    """One judge-accepted synthetic image with boxes."""

    sample_id: str
    anomaly_id: str
    method: str
    source_stem: str
    image_name: str
    boxes: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    seed: int = 0
    judge: dict[str, Any] = field(default_factory=dict)
    variation: dict[str, str] = field(default_factory=dict)
    variation_index: int | None = None


def detections_to_boxes(annotation: AnnotationResult) -> list[dict[str, Any]]:
    return [
        {
            "label": d.label,
            "confidence": float(d.confidence),
            "bbox_xyxy": [float(x) for x in d.bbox_xyxy],
        }
        for d in annotation.detections
    ]


def gt_boxes_to_dicts(boxes: list[DetectionBox]) -> list[dict[str, Any]]:
    return [
        {"label": b.label, "bbox_xyxy": [float(x) for x in b.bbox_xyxy]}
        for b in boxes
    ]


def judge_to_dict(result: JudgeResult) -> dict[str, Any]:
    return {
        "decision": result.decision,
        "overall": float(result.overall),
        "prompt_faithfulness": float(result.prompt_faithfulness),
        "physical_plausibility": float(result.physical_plausibility),
        "annotation_correctness": float(result.annotation_correctness),
        "edge_case_present": bool(result.edge_case_present),
        "rationale": result.rationale,
    }


def save_accepted_image(
    image: Image.Image,
    *,
    out_dir: Path,
    image_name: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / image_name
    image.save(path, quality=92)
    return path


def export_nb2_dataset(
    *,
    output_dir: Path | str,
    accepted: list[AcceptedSample],
    train_real: dict[str, list[Path]],
    test: dict[str, list[Path]],
    real_labels: dict[str, list[DetectionBox]],
    run_stats: dict[str, ClassRunStats],
    config_snapshot: dict[str, Any],
    copy_real_images: bool = False,
) -> dict[str, Path]:
    """Write synth images, combined labels, manifests, and stats under ``output_dir``."""
    root = Path(output_dir)
    synth_dir = root / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[dict[str, Any]]] = {}
    for sample in accepted:
        labels[sample.image_name] = sample.boxes

    # Real train/test manifests (paths relative to original samples_dir).
    def _manifest(groups: dict[str, list[Path]]) -> list[dict[str, Any]]:
        rows = []
        for tag, paths in sorted(groups.items()):
            for path in paths:
                rows.append(
                    {
                        "file": path.name,
                        "tag": tag,
                        "path": str(path),
                        "split": "real",
                        "boxes": gt_boxes_to_dicts(
                            real_labels.get(path.name) or real_labels.get(path.stem) or []
                        ),
                    }
                )
        return rows

    train_manifest = _manifest(train_real)
    test_manifest = _manifest(test)
    for sample in accepted:
        train_manifest.append(
            {
                "file": sample.image_name,
                "tag": sample.anomaly_id,
                "path": str(synth_dir / sample.image_name),
                "split": "synthetic",
                "anomaly_id": sample.anomaly_id,
                "method": sample.method,
                "source_stem": sample.source_stem,
                "boxes": sample.boxes,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "judge": sample.judge,
                "variation": sample.variation,
                "variation_index": sample.variation_index,
            }
        )

    stats_payload = {
        "per_class": {
            k: {
                **asdict(v),
                "acceptance_rate": v.acceptance_rate,
            }
            for k, v in run_stats.items()
        },
        "n_accepted": len(accepted),
        "config": config_snapshot,
    }

    paths = {
        "root": root,
        "synthetic_dir": synth_dir,
        "labels": write_json(root / "labels_synthetic.json", labels),
        "train_manifest": write_json(root / "train_manifest.json", train_manifest),
        "test_manifest": write_json(root / "test_manifest.json", test_manifest),
        "run_stats": write_json(root / "run_stats.json", stats_payload),
    }

    if copy_real_images:
        real_out = root / "real"
        real_out.mkdir(parents=True, exist_ok=True)
        for groups in (train_real, test):
            for path in sum((list(v) for v in groups.values()), []):
                dest = real_out / path.name
                if not dest.exists():
                    shutil.copy2(path, dest)
        paths["real_dir"] = real_out

    return paths


def record_generation(
    *,
    generated: GenerationResult,
    annotation: AnnotationResult,
    judgment: JudgeResult,
    anomaly_id: str,
    method: str,
    source_stem: str,
    image_name: str,
) -> AcceptedSample:
    return AcceptedSample(
        sample_id=Path(image_name).stem,
        anomaly_id=anomaly_id,
        method=method,
        source_stem=source_stem,
        image_name=image_name,
        boxes=detections_to_boxes(annotation),
        prompt=generated.prompt,
        seed=int(generated.seed),
        judge=judge_to_dict(judgment),
        variation=dict(generated.variation or {}),
        variation_index=generated.variation_index,
    )
