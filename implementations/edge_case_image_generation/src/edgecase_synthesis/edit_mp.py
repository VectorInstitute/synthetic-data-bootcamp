"""Process-based multi-GPU edit workers for Notebook 2.

FLUX.2 Klein does **not** support paired img2img batching: a list of images is
treated as shared multi-reference context for every prompt (see Diffusers
discussion #13431). True throughput on 2×L4 needs one process per GPU instead
of ``pipe(prompt=[...], image=[...])`` or threaded dual loads in one process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EditJob:
    """Picklable edit request (no PIL / tensors)."""

    job_id: int
    anomaly_id: str
    method: str
    source_path: str
    source_stem: str
    attempt: int
    variation_index: int


def _box_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> Any:
    import numpy as np

    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
    return mask


def annotation_from_payload(payload: dict[str, Any], image_size: tuple[int, int]) -> Any:
    from edgecase_synthesis.annotation import AnnotationResult, Detection
    import numpy as np

    w, h = image_size
    detections = []
    for d in payload.get("detections") or []:
        bbox = tuple(int(v) for v in d["bbox_xyxy"])
        detections.append(
            Detection(
                label=str(d["label"]),
                confidence=float(d["confidence"]),
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                mask=_box_mask((h, w), bbox),  # type: ignore[arg-type]
            )
        )
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    return AnnotationResult(
        detections=detections,
        overlay=overlay,
        num_instances=len(detections),
    )


def _annotation_to_payload(annotation: Any) -> dict[str, Any]:
    return {
        "detections": [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "bbox_xyxy": [int(x) for x in d.bbox_xyxy],
            }
            for d in annotation.detections
        ]
    }


def mp_synthesize_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker entrypoint: isolate one GPU via CUDA_VISIBLE_DEVICES, then edit.

    Must set the env var **before** importing torch / diffusers.
    """
    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Avoid parent progress-bar spam in workers.
    os.environ["EDGECASE_DISABLE_PIPE_PROGRESS"] = "1"

    from pathlib import Path as _Path

    from PIL import Image

    from edgecase_synthesis.batch_runner import _annotate_item, _load_edit_stack
    from edgecase_synthesis.config import load_config
    from edgecase_synthesis.pipeline import synthesize_one

    project_root = _Path(payload["project_root"])
    cfg = load_config(
        start=project_root,
        overrides=[
            f"dataset_name={payload['dataset_name']}",
            f"hardware={payload['hardware']}",
        ],
    )
    # Only one GPU is visible via CUDA_VISIBLE_DEVICES. device=None →
    # torch.device("cuda") with index=None → enable_model_cpu_offload (fast path).
    jobs = [EditJob(**j) for j in payload["jobs"]]
    warm = set(payload.get("warm_methods") or [])
    stack = _load_edit_stack(
        cfg,
        device=None,
        need_depth=bool(payload.get("need_depth")),
        need_seg=bool(payload.get("need_seg")),
        warm_methods=warm,
        log=lambda msg: print(f"[gpu{gpu_id}] {msg}", flush=True),
    )
    tmp_dir = _Path(payload["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_classes = list(payload.get("base_classes") or [])
    out: list[dict[str, Any]] = []

    for job in jobs:
        image = Image.open(job.source_path).convert("RGB")
        print(
            f"[gpu{gpu_id}] edit {job.anomaly_id} seed={job.source_stem} "
            f"attempt={job.attempt}",
            flush=True,
        )
        depth = (
            stack.depth_model.predict(image) if stack.depth_model is not None else None
        )
        seg = stack.segmenter.predict(image) if stack.segmenter is not None else None
        syn = synthesize_one(
            image,
            anomaly_id=job.anomaly_id,
            method=job.method,
            cfg=cfg,
            comparer=stack.comparer,
            depth=depth,
            segmentation=seg,
            project_root=project_root,
            seed_offset=job.attempt,
            variation_index=job.variation_index,
        )
        # Build a lightweight PendingItem-like object for annotate helper.
        from edgecase_synthesis.batch_runner import PendingItem

        item = PendingItem(
            anomaly_id=job.anomaly_id,
            method=job.method,
            source_path=_Path(job.source_path),
            source_image=image,
            source_stem=job.source_stem,
            attempt=job.attempt,
            variation_index=job.variation_index,
            generated=syn.generated,
        )
        annotation = _annotate_item(
            item,
            annotator=stack.annotator,
            cfg=cfg,
            project_root=project_root,
            base_classes=base_classes,
        )
        out_path = tmp_dir / f"gpu{gpu_id}_job{job.job_id}.png"
        syn.generated.image.save(out_path)
        edit_mask = getattr(syn.generated, "edit_mask", None)
        mask_path = None
        if edit_mask is not None:
            import numpy as np

            mask_path = str(tmp_dir / f"gpu{gpu_id}_job{job.job_id}_mask.npy")
            np.save(mask_path, edit_mask)
        out.append(
            {
                "job_id": job.job_id,
                "image_path": str(out_path),
                "prompt": syn.generated.prompt,
                "negative_prompt": syn.generated.negative_prompt,
                "seed": int(syn.generated.seed),
                "anomaly_id": syn.generated.anomaly_id,
                "method": syn.generated.method,
                "variation": dict(syn.generated.variation or {}),
                "variation_index": syn.generated.variation_index,
                "edit_mask_path": mask_path,
                "annotation": _annotation_to_payload(annotation),
            }
        )

    # Free VRAM before process exit.
    for obj in (stack.comparer, stack.annotator, stack.depth_model, stack.segmenter):
        if obj is not None and hasattr(obj, "unload"):
            obj.unload()
    return out


def apply_mp_results_to_items(
    items: list[Any],
    results: list[dict[str, Any]],
) -> None:
    """Merge worker payloads back onto PendingItem list (by job_id order)."""
    from PIL import Image

    from edgecase_synthesis.generation import GenerationResult

    by_id = {int(r["job_id"]): r for r in results}
    for item in items:
        job_id = int(getattr(item, "_job_id"))
        payload = by_id[job_id]
        image = Image.open(payload["image_path"]).convert("RGB")
        edit_mask = None
        if payload.get("edit_mask_path"):
            import numpy as np

            edit_mask = np.load(payload["edit_mask_path"])
        item.generated = GenerationResult(
            image=image,
            prompt=str(payload.get("prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            seed=int(payload.get("seed") or 0),
            edit_mask=edit_mask,
            anomaly_id=payload.get("anomaly_id"),
            method=payload.get("method"),
            variation=dict(payload.get("variation") or {}) or None,
            variation_index=payload.get("variation_index"),
        )
        item.annotation = annotation_from_payload(
            payload.get("annotation") or {},
            image.size,
        )
