"""Load-once batch synthesis: edit → annotate → judge → retry (parameterized)."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from edgecase_synthesis.annotation import AnnotationResult, OpenVocabAnnotator
from edgecase_synthesis.batch_export import (
    AcceptedSample,
    ClassRunStats,
    record_generation,
    save_accepted_image,
)
from edgecase_synthesis.compare_methods import MethodComparer
from edgecase_synthesis.conditioning import DepthEstimator, Segmenter
from edgecase_synthesis.config import load_anomaly
from edgecase_synthesis.judge import VLMJudge, summarize_annotations
from edgecase_synthesis.pipeline import synthesize_one


@dataclass
class PendingItem:
    """One seed × anomaly still in the batch pipeline."""

    anomaly_id: str
    method: str
    source_path: Path
    source_image: Image.Image
    source_stem: str
    attempt: int = 0
    generated: Any | None = None
    annotation: AnnotationResult | None = None


@dataclass
class BatchResult:
    accepted: list[AcceptedSample] = field(default_factory=list)
    stats: dict[str, ClassRunStats] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)


def _unload(*objs: Any) -> None:
    for obj in objs:
        if obj is not None and hasattr(obj, "unload"):
            obj.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _annotate_item(
    item: PendingItem,
    *,
    annotator: OpenVocabAnnotator,
    cfg: Any,
    project_root: Path,
    base_classes: list[str],
) -> AnnotationResult:
    from edgecase_synthesis.annotation import target_label_names

    anomaly_cfg = load_anomaly(str(cfg.dataset_name), item.anomaly_id, start=project_root)
    anomaly_classes = list(anomaly_cfg.get("annotation_classes", []) or [])
    classes = list(dict.fromkeys([*(anomaly_classes or base_classes)]))
    conf = anomaly_cfg.get("annotation_conf")
    conf = float(conf) if conf is not None else None
    assert item.generated is not None
    targets = target_label_names(item.anomaly_id, anomaly_classes)
    return annotator.annotate(
        item.generated.image,
        classes=classes,
        conf=conf,
        seed_mask=getattr(item.generated, "edit_mask", None),
        seed_label=item.anomaly_id,
        target_labels=targets,
    )


def run_batch_synthesis(
    seeds_by_anomaly: dict[str, list[Path]],
    method_map: dict[str, str],
    *,
    cfg: Any,
    project_root: Path,
    synth_dir: Path,
    max_retries: int = 2,
    target_accepts: dict[str, int] | None = None,
    require_target_boxes: bool = True,
    progress: Callable[[str], None] | None = print,
) -> BatchResult:
    """Generate, annotate, and judge many seeds with models loaded once per phase.

    Phases (VRAM-friendly on L4):
      1. Load depth + segmenter + comparer + annotator → synthesize + annotate
      2. Unload edit stack → load judge → decide accept / retry / reject
      3. On retries: reload edit stack, re-edit failed items, re-judge

    When ``require_target_boxes`` is True (default), an item cannot be accepted
    without at least one target-class box (YOLO-World or edit-mask fallback).
    """
    from edgecase_synthesis.annotation import (
        check_box_placement,
        has_target_detections,
        target_label_names,
    )

    log = progress or (lambda _msg: None)
    dataset = str(cfg.dataset_name)
    source_hint = str(cfg.dataset.get("source_hint", "a real photograph"))
    base_classes = list(cfg.annotation.classes)
    synth_dir = Path(synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)

    result = BatchResult(
        stats={aid: ClassRunStats(anomaly_id=aid) for aid in seeds_by_anomaly}
    )
    accepted_counts = {aid: 0 for aid in seeds_by_anomaly}

    # Build work queue from caller-provided seeds (notebook chooses counts).
    queue: list[PendingItem] = []
    for anomaly_id, paths in seeds_by_anomaly.items():
        method = method_map[anomaly_id]
        for path in paths:
            image = Image.open(path).convert("RGB")
            queue.append(
                PendingItem(
                    anomaly_id=anomaly_id,
                    method=method,
                    source_path=path,
                    source_image=image,
                    source_stem=path.stem,
                )
            )

    # Per-class counter into the shuffled variation cycle (retries advance too).
    variation_counters: dict[str, int] = {aid: 0 for aid in seeds_by_anomaly}

    if not queue:
        log("No seeds queued — nothing to synthesize.")
        return result

    depth_model: DepthEstimator | None = None
    segmenter: Segmenter | None = None
    comparer: MethodComparer | None = None
    annotator: OpenVocabAnnotator | None = None
    judge: VLMJudge | None = None

    def load_edit_stack() -> None:
        nonlocal depth_model, segmenter, comparer, annotator, judge
        _unload(judge)
        judge = None
        if depth_model is None:
            depth_model = DepthEstimator.from_config(cfg)
        if segmenter is None:
            segmenter = Segmenter.from_config(cfg)
        if comparer is None:
            comparer = MethodComparer.from_config(cfg)
        if annotator is None:
            annotator = OpenVocabAnnotator.from_config(cfg)

    def load_judge() -> None:
        nonlocal depth_model, segmenter, comparer, annotator, judge
        _unload(comparer, annotator, depth_model, segmenter)
        comparer = annotator = depth_model = segmenter = None
        if judge is None:
            judge = VLMJudge.from_config(cfg)

    def synthesize_queue(items: list[PendingItem]) -> None:
        assert depth_model and segmenter and comparer and annotator
        for item in items:
            # Skip if this class already hit its accept target.
            if target_accepts:
                want = int(target_accepts.get(item.anomaly_id, 10**9))
                if accepted_counts[item.anomaly_id] >= want:
                    continue
            log(
                f"  edit  {item.anomaly_id}  seed={item.source_stem}  "
                f"attempt={item.attempt}  method={item.method}"
            )
            depth = depth_model.predict(item.source_image)
            seg = segmenter.predict(item.source_image)
            var_idx = variation_counters[item.anomaly_id]
            variation_counters[item.anomaly_id] = var_idx + 1
            syn = synthesize_one(
                item.source_image,
                anomaly_id=item.anomaly_id,
                method=item.method,
                cfg=cfg,
                comparer=comparer,
                depth=depth,
                segmentation=seg,
                project_root=project_root,
                seed_offset=item.attempt,
                variation_index=var_idx,
            )
            item.generated = syn.generated
            if syn.generated.variation:
                log(
                    f"    variation[{var_idx}] "
                    + ", ".join(f"{k}={v}" for k, v in syn.generated.variation.items())
                )
            item.annotation = _annotate_item(
                item,
                annotator=annotator,
                cfg=cfg,
                project_root=project_root,
                base_classes=base_classes,
            )
            result.stats[item.anomaly_id].attempts += 1

    def judge_queue(items: list[PendingItem]) -> list[PendingItem]:
        """Judge items; return those that should retry."""
        assert judge is not None
        retries: list[PendingItem] = []
        for item in items:
            if item.generated is None or item.annotation is None:
                continue
            if target_accepts:
                want = int(target_accepts.get(item.anomaly_id, 10**9))
                if accepted_counts[item.anomaly_id] >= want:
                    continue
            anomaly_cfg = load_anomaly(dataset, item.anomaly_id, start=project_root)
            anomaly_classes = list(anomaly_cfg.get("annotation_classes", []) or [])
            targets = target_label_names(item.anomaly_id, anomaly_classes)
            boxed = has_target_detections(item.annotation, targets)

            judgment = judge.judge(
                item.generated.image,
                prompt=item.generated.prompt,
                anomaly_id=item.anomaly_id,
                anomaly_name=str(anomaly_cfg.get("display_name", item.anomaly_id)),
                annotations_summary=summarize_annotations(item.annotation),
                source_hint=source_hint,
                require_target_boxes=require_target_boxes,
                has_target_boxes=boxed,
            )
            decision = judgment.decision
            if require_target_boxes and not boxed:
                # Hard rule: no target box → never accept into the training export.
                if item.attempt < max_retries:
                    decision = "retry"
                else:
                    decision = "reject"
                note = (
                    f" [box-gate: no target box → {decision} "
                    f"(open-vocab + edit-mask fallback both empty)]"
                )
                judgment.decision = decision
                judgment.rationale = (judgment.rationale or "").rstrip() + note

            gates = dict(anomaly_cfg.get("accept_gates") or {})
            if decision == "accept" and gates and boxed:
                w, h = item.generated.image.size
                ok, reason = check_box_placement(
                    item.annotation,
                    targets,
                    image_size=(w, h),
                    gates=gates,
                )
                if not ok:
                    if item.attempt < max_retries:
                        decision = "retry"
                    else:
                        decision = "reject"
                    note = f" [placement-gate: {reason} → {decision}]"
                    judgment.decision = decision
                    judgment.rationale = (judgment.rationale or "").rstrip() + note

            log(
                f"  judge {item.anomaly_id}  seed={item.source_stem}  "
                f"attempt={item.attempt} → {decision} "
                f"({judgment.overall:.1f})  boxes={'yes' if boxed else 'NO'}"
            )
            stats = result.stats[item.anomaly_id]
            if decision == "accept":
                stats.accepts += 1
                accepted_counts[item.anomaly_id] += 1
                image_name = (
                    f"synth_{item.anomaly_id}_{item.source_stem}_a{item.attempt}.jpg"
                )
                save_accepted_image(item.generated.image, out_dir=synth_dir, image_name=image_name)
                result.accepted.append(
                    record_generation(
                        generated=item.generated,
                        annotation=item.annotation,
                        judgment=judgment,
                        anomaly_id=item.anomaly_id,
                        method=item.method,
                        source_stem=item.source_stem,
                        image_name=image_name,
                    )
                )
            elif decision == "retry" and item.attempt < max_retries:
                stats.retries += 1
                item.attempt += 1
                item.generated = None
                item.annotation = None
                retries.append(item)
            else:
                stats.rejects += 1
                result.rejected.append(
                    {
                        "anomaly_id": item.anomaly_id,
                        "source_stem": item.source_stem,
                        "attempt": item.attempt,
                        "decision": decision,
                        "overall": float(judgment.overall),
                        "has_target_boxes": boxed,
                    }
                )
        return retries

    # --- phase 1: first-pass edits ---
    log(f"Loading edit stack once ({len(queue)} jobs)…")
    load_edit_stack()
    active = list(queue)
    synthesize_queue(active)

    # --- phase 2+: judge / retry loop ---
    while active:
        log("Switching to judge…")
        load_judge()
        retries = judge_queue(active)
        if not retries:
            break
        # Drop classes that already met target.
        if target_accepts:
            retries = [
                r
                for r in retries
                if accepted_counts[r.anomaly_id] < int(target_accepts.get(r.anomaly_id, 10**9))
            ]
        if not retries:
            break
        log(f"Retrying {len(retries)} item(s)…")
        load_edit_stack()
        synthesize_queue(retries)
        active = retries

    _unload(comparer, annotator, depth_model, segmenter, judge)
    log(
        f"Done. Accepted {len(result.accepted)} / "
        f"{sum(s.attempts for s in result.stats.values())} attempts."
    )
    return result
