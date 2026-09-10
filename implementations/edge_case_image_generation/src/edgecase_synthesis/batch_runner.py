"""Load-once batch synthesis: edit → annotate → judge → retry (parameterized).

Speed path on ``gpu_l4x2`` (prefer ``scripts/run_nb2_batch.py``):
  - Skip depth/seg when every queued method is instruct-only (NB2 default).
  - One Klein process per GPU (``edit_mp``) — not Diffusers list-batch (Klein
    treats ``image=[...]`` as multi-ref for every prompt, not paired rows).
  - Fan out API judge calls with a thread pool (I/O bound).
  - Checkpoint accepts/rejects under ``nb2/checkpoint/`` for ``--resume``.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from edgecase_synthesis.annotation import AnnotationResult, OpenVocabAnnotator
from edgecase_synthesis.batch_checkpoint import (
    accepted_path,
    append_jsonl,
    accepted_sample_to_row,
    load_checkpoint,
    rebuild_stats,
    rejected_path,
    sample_key,
    save_state,
)
from edgecase_synthesis.batch_export import (
    AcceptedSample,
    ClassRunStats,
    record_generation,
    save_accepted_image,
)
from edgecase_synthesis.compare_methods import METHOD_SPECS, MethodComparer
from edgecase_synthesis.conditioning import DepthEstimator, Segmenter
from edgecase_synthesis.config import load_anomaly
from edgecase_synthesis.judge import JudgeResult, VLMJudge, summarize_annotations
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
    variation_index: int | None = None
    generated: Any | None = None
    annotation: AnnotationResult | None = None


@dataclass
class BatchResult:
    accepted: list[AcceptedSample] = field(default_factory=list)
    stats: dict[str, ClassRunStats] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _EditStack:
    device: str
    depth_model: DepthEstimator | None
    segmenter: Segmenter | None
    comparer: MethodComparer
    annotator: OpenVocabAnnotator


def _unload(*objs: Any) -> None:
    for obj in objs:
        if obj is not None and hasattr(obj, "unload"):
            obj.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hw_int(cfg: Any, key: str, default: int) -> int:
    hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
    if hardware is None:
        return int(default)
    raw = hardware.get(key)
    if raw in (None, ""):
        return int(default)
    return int(raw)


def _in_ipython() -> bool:
    try:
        get_ipython()  # type: ignore[name-defined]  # noqa: F821
        return True
    except NameError:
        return False


def _judge_workers(cfg: Any) -> int:
    judge_cfg = cfg.get("judge") if hasattr(cfg, "get") else None
    backend = ""
    if judge_cfg is not None:
        backend = str(judge_cfg.get("backend") or "").lower()
        raw = judge_cfg.get("max_parallel")
        if raw not in (None, ""):
            n = max(1, int(raw))
        else:
            n = max(1, _hw_int(cfg, "judge_workers", 4))
    else:
        n = max(1, _hw_int(cfg, "judge_workers", 4))
    # Local CUDA judges are not thread-safe / waste VRAM when fanned out.
    if backend in {"qwen_vl", "clip"}:
        return 1
    return n


def _edit_workers(cfg: Any, *, log: Callable[[str], None] | None = None) -> int:
    # Process-based multi-GPU (see edit_mp). Default follows num_gpus when set.
    requested = _hw_int(cfg, "parallel_edit_workers", 0)
    if requested <= 0:
        requested = _hw_int(cfg, "num_gpus", 1)
    if not torch.cuda.is_available():
        return 1
    n = max(1, min(int(requested), int(torch.cuda.device_count())))
    # Jupyter + ProcessPool + CUDA often hangs on interrupt; dual-GPU belongs in CLI.
    allow_nb_mp = os.environ.get("EDGECASE_ALLOW_NOTEBOOK_MP", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if n > 1 and _in_ipython() and not allow_nb_mp:
        if log:
            log(
                "Notebook detected: capping edit_workers=1 (ProcessPool+CUDA is flaky "
                "in Jupyter). Use scripts/run_nb2_batch.py for dual-GPU, or set "
                "EDGECASE_ALLOW_NOTEBOOK_MP=1 to override."
            )
        return 1
    return n


def _methods_need_conditioning(methods: set[str]) -> tuple[bool, bool]:
    need_depth = False
    need_seg = False
    for method in methods:
        spec = METHOD_SPECS.get(str(method).lower())
        if spec is None:
            need_depth = need_seg = True
            break
        need_depth = need_depth or bool(spec.uses_depth or spec.uses_mask)
        need_seg = need_seg or bool(spec.uses_seg or spec.uses_mask)
    return need_depth, need_seg


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


def _load_edit_stack(
    cfg: Any,
    *,
    device: str | None,
    need_depth: bool,
    need_seg: bool,
    warm_methods: set[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> _EditStack:
    depth_model = DepthEstimator.from_config(cfg, device=device) if need_depth else None
    segmenter = Segmenter.from_config(cfg, device=device) if need_seg else None
    comparer = MethodComparer.from_config(cfg, device=device)
    annotator = OpenVocabAnnotator.from_config(cfg, device=device)
    stack = _EditStack(
        device=str(device or "auto"),
        depth_model=depth_model,
        segmenter=segmenter,
        comparer=comparer,
        annotator=annotator,
    )
    # Build Klein pipes here (sequentially per GPU) so parallel workers don't
    # race two from_pretrained calls and hit meta-tensor placement bugs.
    warm = warm_methods or set()
    if "instruct" in warm and comparer.instruct_is_klein:
        if log:
            log(f"  warming Klein instruct on {stack.device}…")
        _ = comparer.instruct_pipe
    if "inpaint" in warm and comparer.inpaint_is_klein:
        if log:
            log(f"  warming Klein inpaint on {stack.device}…")
        _ = comparer.inpaint_pipe
    return stack


def _synthesize_items(
    items: list[PendingItem],
    *,
    stack: _EditStack,
    cfg: Any,
    project_root: Path,
    base_classes: list[str],
    target_accepts: dict[str, int] | None,
    accepted_counts: dict[str, int],
    stats: dict[str, ClassRunStats],
    log: Callable[[str], None],
    stats_lock: threading.Lock | None = None,
    show_progress: bool = True,
) -> None:
    from tqdm.auto import tqdm

    iterator: Any = items
    if show_progress:
        iterator = tqdm(items, desc="Editing", unit="img", leave=True)

    for item in iterator:
        if target_accepts:
            want = int(target_accepts.get(item.anomaly_id, 10**9))
            if accepted_counts[item.anomaly_id] >= want:
                continue
        log(
            f"  edit  {item.anomaly_id}  seed={item.source_stem}  "
            f"attempt={item.attempt}  method={item.method}  device={stack.device}"
        )
        depth = (
            stack.depth_model.predict(item.source_image) if stack.depth_model is not None else None
        )
        seg = stack.segmenter.predict(item.source_image) if stack.segmenter is not None else None
        var_idx = int(item.variation_index or 0)
        try:
            syn = synthesize_one(
                item.source_image,
                anomaly_id=item.anomaly_id,
                method=item.method,
                cfg=cfg,
                comparer=stack.comparer,
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
                annotator=stack.annotator,
                cfg=cfg,
                project_root=project_root,
                base_classes=base_classes,
            )
            if stats_lock is None:
                stats[item.anomaly_id].attempts += 1
            else:
                with stats_lock:
                    stats[item.anomaly_id].attempts += 1
        except Exception as exc:  # noqa: BLE001 — keep batch alive
            item.generated = None
            item.annotation = None
            setattr(item, "_edit_error", f"{type(exc).__name__}: {exc}")
            log(f"    FAILED: {exc}")
            if stats_lock is None:
                stats[item.anomaly_id].attempts += 1
            else:
                with stats_lock:
                    stats[item.anomaly_id].attempts += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _resolve_progress(
    progress: Any,
    *,
    verbose: bool,
) -> tuple[Callable[[str], None], Callable[[str], None], bool]:
    """Return ``(log_phase, log_item, use_tqdm)``."""
    from tqdm.auto import tqdm

    if callable(progress) and progress not in (print,) and progress != "tqdm":
        # Custom callable: treat as phase logger; item lines follow verbose.
        log_phase: Callable[[str], None] = progress
        log_item: Callable[[str], None] = progress if verbose else (lambda _m: None)
        return log_phase, log_item, False

    mode = progress
    if mode is print:
        mode = "print"
    if mode is None:
        mode = "tqdm"
    if mode == "silent":
        return (lambda _m: None), (lambda _m: None), False
    if mode == "print":
        return print, (print if verbose else (lambda _m: None)), False
    # default: tqdm
    return tqdm.write, (tqdm.write if verbose else (lambda _m: None)), True


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
    progress: Any = "tqdm",
    verbose: bool = False,
    resume: bool = False,
    nb2_dir: Path | str | None = None,
) -> BatchResult:
    """Generate, annotate, and judge many seeds with models loaded once per phase.

    Phases:
      1. Load edit stack(s) → synthesize + annotate (optionally 1 stack per GPU)
      2. Unload edit stack → load judge → concurrent API decisions
      3. On retries: reload edit stack(s), re-edit, re-judge

    When ``require_target_boxes`` is True (default), an item cannot be accepted
    without at least one target-class box (YOLO-World or edit-mask fallback).

    Set ``resume=True`` to skip ``(anomaly, seed)`` pairs already accepted or
    finally rejected under ``<nb2_dir>/checkpoint/``.

    ``progress``: ``\"tqdm\"`` (default), ``\"print\"``, ``\"silent\"``, or a callable.
    Per-item lines are off unless ``verbose=True``.
    """
    from edgecase_synthesis.annotation import (
        check_box_placement,
        has_target_detections,
        target_label_names,
    )
    from tqdm.auto import tqdm

    log_phase, log_item, use_tqdm = _resolve_progress(progress, verbose=verbose)
    log = log_phase  # phase / summary messages
    dataset = str(cfg.dataset_name)
    source_hint = str(cfg.dataset.get("source_hint", "a real photograph"))
    base_classes = list(cfg.annotation.classes)
    synth_dir = Path(synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(nb2_dir) if nb2_dir is not None else synth_dir.parent
    root_dir.mkdir(parents=True, exist_ok=True)

    result = BatchResult(
        stats={aid: ClassRunStats(anomaly_id=aid) for aid in seeds_by_anomaly}
    )
    accepted_counts = {aid: 0 for aid in seeds_by_anomaly}
    done_keys: set[str] = set()
    variation_counters: dict[str, int] = {aid: 0 for aid in seeds_by_anomaly}

    if resume:
        ckpt = load_checkpoint(root_dir)
        result.accepted = list(ckpt["accepted"])
        result.rejected = list(ckpt["rejected"])
        done_keys = set(ckpt["accepted_keys"]) | set(ckpt["rejected_keys"])
        for sample in result.accepted:
            accepted_counts[sample.anomaly_id] = accepted_counts.get(sample.anomaly_id, 0) + 1
        saved_vars = (ckpt.get("state") or {}).get("variation_counters") or {}
        for aid, val in saved_vars.items():
            variation_counters[str(aid)] = int(val)
        result.stats = rebuild_stats(
            list(seeds_by_anomaly.keys()),
            accepted=result.accepted,
            rejected=result.rejected,
            state=ckpt.get("state"),
        )
        log(
            f"Resume: loaded {len(result.accepted)} accepted, "
            f"{len(result.rejected)} rejected; skipping {len(done_keys)} seeds."
        )

    queue: list[PendingItem] = []
    for anomaly_id, paths in seeds_by_anomaly.items():
        method = method_map[anomaly_id]
        for path in paths:
            key = sample_key(anomaly_id, path.stem)
            if key in done_keys:
                continue
            if target_accepts and accepted_counts.get(anomaly_id, 0) >= int(
                target_accepts.get(anomaly_id, 10**9)
            ):
                continue
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

    if not queue:
        log("No seeds queued — nothing to synthesize (targets met or all done).")
        return result

    n_edit = _edit_workers(cfg, log=log)
    n_judge = _judge_workers(cfg)
    methods_in_queue = {item.method for item in queue}
    need_depth, need_seg = _methods_need_conditioning(methods_in_queue)
    instruct_id = str(cfg.generation.get("instruct_model_id") or "")
    inpaint_id = str(cfg.generation.get("inpaint_model_id") or "")
    needs_klein = (
        ("instruct" in methods_in_queue and MethodComparer._is_klein_model(instruct_id))
        or ("inpaint" in methods_in_queue and MethodComparer._is_klein_model(inpaint_id))
    )
    if needs_klein:
        from edgecase_synthesis.diffusers_klein import assert_klein_available

        assert_klein_available()
    log(
        f"Batch parallelism: edit_workers={n_edit}  judge_workers={n_judge}  "
        f"depth={need_depth}  seg={need_seg}"
    )

    stacks: list[_EditStack] = []
    judge: VLMJudge | None = None

    def _persist_state(pending: list[PendingItem]) -> None:
        save_state(
            root_dir,
            {
                "variation_counters": dict(variation_counters),
                "stats": {k: asdict(v) for k, v in result.stats.items()},
                "pending": [
                    {
                        "anomaly_id": p.anomaly_id,
                        "source_stem": p.source_stem,
                        "source_path": str(p.source_path),
                        "method": p.method,
                        "attempt": int(p.attempt),
                        "variation_index": p.variation_index,
                    }
                    for p in pending
                ],
            },
        )

    def unload_edit_stacks() -> None:
        nonlocal stacks
        for stack in stacks:
            _unload(stack.comparer, stack.annotator, stack.depth_model, stack.segmenter)
        stacks = []

    def load_edit_stacks() -> None:
        nonlocal stacks, judge
        _unload(judge)
        judge = None
        unload_edit_stacks()
        if n_edit > 1:
            log(
                f"Edit phase: {n_edit} process workers (one Klein per GPU). "
                "Note: Flux2Klein list(image)+list(prompt) is multi-ref, not paired batch."
            )
            os.environ.pop("EDGECASE_DISABLE_PIPE_PROGRESS", None)
            return
        os.environ.pop("EDGECASE_DISABLE_PIPE_PROGRESS", None)
        log("Loading edit stack on default device…")
        stacks.append(
            _load_edit_stack(
                cfg,
                device=None,
                need_depth=need_depth,
                need_seg=need_seg,
                warm_methods=methods_in_queue,
                log=log,
            )
        )

    def load_judge() -> None:
        nonlocal judge
        unload_edit_stacks()
        if judge is None:
            judge = VLMJudge.from_config(cfg)
            # Resume: re-seed novelty bank from already-accepted synths.
            if resume and result.accepted:
                for sample in result.accepted:
                    path = synth_dir / sample.image_name
                    if not path.exists():
                        continue
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception:
                        continue
                    judge.register_accepted_embedding(
                        sample.anomaly_id, img, path=str(path)
                    )
                log(
                    f"Resume: seeded embedding novelty bank with "
                    f"{len(result.accepted)} accepted image(s)."
                )

    def assign_variations(items: list[PendingItem]) -> list[PendingItem]:
        active: list[PendingItem] = []
        for item in items:
            if target_accepts:
                want = int(target_accepts.get(item.anomaly_id, 10**9))
                if accepted_counts[item.anomaly_id] >= want:
                    continue
            item.variation_index = variation_counters[item.anomaly_id]
            variation_counters[item.anomaly_id] = int(item.variation_index) + 1
            active.append(item)
        return active

    def synthesize_queue(items: list[PendingItem]) -> None:
        active = assign_variations(items)
        if not active:
            return
        if n_edit <= 1:
            if not stacks:
                load_edit_stacks()
            assert stacks
            _synthesize_items(
                active,
                stack=stacks[0],
                cfg=cfg,
                project_root=project_root,
                base_classes=base_classes,
                target_accepts=target_accepts,
                accepted_counts=accepted_counts,
                stats=result.stats,
                log=log_item,
                show_progress=use_tqdm,
            )
            return

        from concurrent.futures import ProcessPoolExecutor

        from edgecase_synthesis.edit_mp import (
            EditJob,
            apply_mp_results_to_items,
            mp_synthesize_shard,
        )

        unload_edit_stacks()
        hardware_name = str(cfg.hardware.get("name") or cfg.get("hardware") or "gpu_l4")
        tmp_root = Path(tempfile.mkdtemp(prefix="edgecase_edit_"))
        try:
            for i, item in enumerate(active):
                item._job_id = i  # type: ignore[attr-defined]
            shards = [active[i::n_edit] for i in range(n_edit)]
            payloads = []
            for gpu_id, shard in enumerate(shards):
                if not shard:
                    continue
                jobs = [
                    EditJob(
                        job_id=int(item._job_id),  # type: ignore[attr-defined]
                        anomaly_id=item.anomaly_id,
                        method=item.method,
                        source_path=str(item.source_path),
                        source_stem=item.source_stem,
                        attempt=int(item.attempt),
                        variation_index=int(item.variation_index or 0),
                    )
                    for item in shard
                ]
                payloads.append(
                    {
                        "gpu_id": gpu_id,
                        "jobs": [j.__dict__ for j in jobs],
                        "project_root": str(project_root),
                        "dataset_name": dataset,
                        "hardware": hardware_name,
                        "need_depth": need_depth,
                        "need_seg": need_seg,
                        "warm_methods": list(methods_in_queue),
                        "base_classes": base_classes,
                        "tmp_dir": str(tmp_root / f"gpu{gpu_id}"),
                    }
                )
            log(f"Dispatching {len(active)} edits across {len(payloads)} GPU processes…")
            ctx = __import__("multiprocessing").get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as pool:
                shard_results = list(pool.map(mp_synthesize_shard, payloads))
            flat: list[dict] = []
            for part in shard_results:
                flat.extend(part)
            failed = apply_mp_results_to_items(active, flat)
            for item in active:
                if item.generated is not None:
                    result.stats[item.anomaly_id].attempts += 1
            if failed:
                log(f"  {len(failed)} edit(s) failed in workers (will retry/reject).")
            log(f"Process edit done ({len(flat) - len(failed)} ok / {len(flat)} jobs)")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def _judge_api_call(item: PendingItem) -> tuple[PendingItem, JudgeResult, bool]:
        assert judge is not None and item.generated is not None and item.annotation is not None
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
            exclude_stems={item.source_stem},
            edit_mask=getattr(item.generated, "edit_mask", None),
            annotation=item.annotation,
        )
        return item, judgment, boxed

    def judge_queue(items: list[PendingItem]) -> list[PendingItem]:
        """Judge items; return those that should retry."""
        assert judge is not None
        retries: list[PendingItem] = []

        for item in items:
            if item.generated is not None and item.annotation is not None:
                continue
            stats = result.stats[item.anomaly_id]
            err = getattr(item, "_edit_error", None) or "edit produced no image"
            if item.attempt < max_retries:
                stats.retries += 1
                item.attempt += 1
                item.generated = None
                item.annotation = None
                retries.append(item)
                log_item(
                    f"  edit-fail {item.anomaly_id}  seed={item.source_stem}  "
                    f"→ retry ({err})"
                )
            else:
                stats.rejects += 1
                row = {
                    "anomaly_id": item.anomaly_id,
                    "source_stem": item.source_stem,
                    "attempt": item.attempt,
                    "decision": "reject",
                    "overall": 0.0,
                    "has_target_boxes": False,
                    "error": err,
                }
                result.rejected.append(row)
                append_jsonl(rejected_path(root_dir), row)
                done_keys.add(sample_key(item.anomaly_id, item.source_stem))
                log_item(
                    f"  edit-fail {item.anomaly_id}  seed={item.source_stem}  "
                    f"→ reject ({err})"
                )

        ready = [
            item
            for item in items
            if item.generated is not None
            and item.annotation is not None
            and (
                not target_accepts
                or accepted_counts[item.anomaly_id]
                < int(target_accepts.get(item.anomaly_id, 10**9))
            )
        ]
        if not ready:
            return retries

        judged: list[tuple[PendingItem, JudgeResult, bool]] = []
        workers = min(n_judge, len(ready))
        if workers <= 1:
            ready_iter: Any = ready
            if use_tqdm:
                ready_iter = tqdm(ready, desc="Judging", unit="img", leave=True)
            judged = [_judge_api_call(item) for item in ready_iter]
        else:
            log(f"Judging {len(ready)} item(s) with {workers} parallel API calls…")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_judge_api_call, item) for item in ready]
                done_iter: Any = as_completed(futs)
                if use_tqdm:
                    done_iter = tqdm(
                        done_iter,
                        total=len(futs),
                        desc="Judging",
                        unit="img",
                        leave=True,
                    )
                for fut in done_iter:
                    judged.append(fut.result())

        accept_n = retry_n = reject_n = 0
        for item, judgment, boxed in judged:
            anomaly_cfg = load_anomaly(dataset, item.anomaly_id, start=project_root)
            anomaly_classes = list(anomaly_cfg.get("annotation_classes", []) or [])
            targets = target_label_names(item.anomaly_id, anomaly_classes)
            decision = judgment.decision
            if require_target_boxes and not boxed:
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
                assert item.generated is not None and item.annotation is not None
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

            log_item(
                f"  judge {item.anomaly_id}  seed={item.source_stem}  "
                f"attempt={item.attempt} → {decision} "
                f"({judgment.overall:.1f})"
                + (
                    f"  fid={judgment.global_fidelity:.1f}/{judgment.object_fidelity:.1f}"
                    if judgment.global_fidelity is not None
                    and judgment.object_fidelity is not None
                    else ""
                )
                + (
                    f"  emb={judgment.embed_real_sim_global:.2f}/{judgment.embed_neighbor_sim:.2f}"
                    if judgment.embed_real_sim_global is not None
                    and judgment.embed_neighbor_sim is not None
                    else ""
                )
                + f"  boxes={'yes' if boxed else 'NO'}"
            )
            stats = result.stats[item.anomaly_id]
            if decision == "accept":
                accept_n += 1
                stats.accepts += 1
                accepted_counts[item.anomaly_id] += 1
                image_name = (
                    f"synth_{item.anomaly_id}_{item.source_stem}_a{item.attempt}.jpg"
                )
                assert item.generated is not None and item.annotation is not None
                saved = save_accepted_image(
                    item.generated.image, out_dir=synth_dir, image_name=image_name
                )
                # Grow novelty bank so later accepts stay diverse within the run.
                assert judge is not None
                judge.register_accepted_embedding(
                    item.anomaly_id,
                    item.generated.image,
                    path=str(saved),
                )
                sample = record_generation(
                    generated=item.generated,
                    annotation=item.annotation,
                    judgment=judgment,
                    anomaly_id=item.anomaly_id,
                    method=item.method,
                    source_stem=item.source_stem,
                    image_name=image_name,
                )
                result.accepted.append(sample)
                append_jsonl(accepted_path(root_dir), accepted_sample_to_row(sample))
                done_keys.add(sample_key(item.anomaly_id, item.source_stem))
            elif decision == "retry" and item.attempt < max_retries:
                retry_n += 1
                stats.retries += 1
                item.attempt += 1
                item.generated = None
                item.annotation = None
                retries.append(item)
            else:
                reject_n += 1
                stats.rejects += 1
                row = {
                    "anomaly_id": item.anomaly_id,
                    "source_stem": item.source_stem,
                    "attempt": item.attempt,
                    "decision": decision,
                    "overall": float(judgment.overall),
                    "has_target_boxes": boxed,
                }
                result.rejected.append(row)
                append_jsonl(rejected_path(root_dir), row)
                done_keys.add(sample_key(item.anomaly_id, item.source_stem))
        log(
            f"Judge round: accept={accept_n}  retry={retry_n}  reject={reject_n}  "
            f"accepted_total={len(result.accepted)}"
        )
        return retries

    log(f"Loading edit stack once ({len(queue)} jobs)…")
    load_edit_stacks()
    active = list(queue)
    synthesize_queue(active)
    _persist_state(active)

    while active:
        log("Switching to judge…")
        load_judge()
        retries = judge_queue(active)
        _persist_state(retries)
        if not retries:
            break
        if target_accepts:
            retries = [
                r
                for r in retries
                if accepted_counts[r.anomaly_id] < int(target_accepts.get(r.anomaly_id, 10**9))
            ]
        if not retries:
            break
        log(f"Retrying {len(retries)} item(s)…")
        load_edit_stacks()
        synthesize_queue(retries)
        _persist_state(retries)
        active = retries

    unload_edit_stacks()
    _unload(judge)
    os.environ.pop("EDGECASE_DISABLE_PIPE_PROGRESS", None)
    _persist_state([])
    log(
        f"Done. Accepted {len(result.accepted)} / "
        f"{sum(s.attempts for s in result.stats.values())} attempts."
    )
    return result
