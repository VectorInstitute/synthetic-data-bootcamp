"""Load-once batch synthesis: edit → annotate → judge → retry (parameterized).

Speed path on ``gpu_l4x2`` (prefer ``scripts/run_nb2_batch.py``):
  - Skip depth/seg when every queued method is instruct-only (NB2 default).
  - One Klein process per GPU (``edit_mp``) — not Diffusers list-batch (Klein
    treats ``image=[...]`` as multi-ref for every prompt, not paired rows).
  - Fan out API judge calls with a thread pool (I/O bound).
  - Checkpoint accepts/rejects under ``nb2/checkpoint/`` for ``--resume``.

Target semantics: ``target_accepts`` is the number of *accepted* images wanted
per class. With ``max_attempts`` set, classes still short of their target after
the first pass revisit their seed pool (next prompt variation + new diffusion
seed) until the target is met or the per-class attempt budget is spent.
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from tqdm.auto import tqdm

from aieng.syn_data.image.batch.checkpoint import (
    accepted_path,
    accepted_sample_to_row,
    append_jsonl,
    embeddings_path,
    judged_path,
    load_checkpoint,
    load_embedding_artifacts,
    rebuild_stats,
    rejected_path,
    sample_key,
    save_real_embeddings,
    save_state,
)
from aieng.syn_data.image.batch.export import (
    AcceptedSample,
    ClassRunStats,
    judge_to_dict,
    record_generation,
    save_accepted_image,
)
from aieng.syn_data.image.config import load_anomaly
from aieng.syn_data.image.generate.annotation import (
    AnnotationResult,
    OpenVocabAnnotator,
    check_box_placement,
    has_target_detections,
    target_label_names,
)
from aieng.syn_data.image.generate.compare_methods import METHOD_SPECS, MethodComparer
from aieng.syn_data.image.generate.conditioning import DepthEstimator, Segmenter
from aieng.syn_data.image.generate.diffusers_klein import assert_klein_available
from aieng.syn_data.image.generate.pipeline import synthesize_one
from aieng.syn_data.image.judge import JudgeResult, VLMJudge, summarize_annotations


# Diffusion seed stride between seed-pool passes (pass 0 keeps seed_offset=attempt).
_PASS_SEED_STRIDE = 1000
# Floor on the observed acceptance rate when sizing refills (avoids huge requests).
_MIN_REFILL_RATE = 0.1


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
    pass_index: int = 0

    @property
    def seed_offset(self) -> int:
        """Diffusion seed offset; unique per (pass, attempt) for one seed."""
        return int(self.attempt) + _PASS_SEED_STRIDE * int(self.pass_index)

    @property
    def key(self) -> str:
        """Checkpoint key; revisits of a seed get their own key."""
        stem = (
            self.source_stem
            if self.pass_index == 0
            else f"{self.source_stem}@p{self.pass_index}"
        )
        return sample_key(self.anomaly_id, stem)


@dataclass
class BatchResult:
    """Represent BatchResult configuration and behavior.

    ``judged`` holds one row per judged edit (accepted, retried, rejected, or
    surplus) with the judge scores and embedding sims, for plots/diagnostics.
    ``embeddings`` holds the matching CLIP vectors and ``real_embeddings`` the
    real same-class bank they were compared to (per class).
    ``surplus`` counts edits that passed every gate after their class had
    already reached its target (not exported).
    """

    accepted: list[AcceptedSample] = field(default_factory=list)
    stats: dict[str, ClassRunStats] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    judged: list[dict[str, Any]] = field(default_factory=list)
    embeddings: list[dict[str, Any]] = field(default_factory=list)
    real_embeddings: dict[str, Any] = field(default_factory=dict)
    surplus: dict[str, int] = field(default_factory=dict)
    targets: dict[str, int] = field(default_factory=dict)
    max_attempts: dict[str, int] = field(default_factory=dict)
    scene_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def accepted_counts(self) -> dict[str, int]:
        """Count accepted images per class."""
        counts = dict.fromkeys(self.stats, 0)
        for sample in self.accepted:
            counts[sample.anomaly_id] = counts.get(sample.anomaly_id, 0) + 1
        return counts

    def target_reached(self) -> dict[str, bool]:
        """Whether each targeted class reached its accepted-image target."""
        counts = self.accepted_counts()
        return {aid: counts.get(aid, 0) >= want for aid, want in self.targets.items()}


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
        n = (
            max(1, int(raw))
            if raw not in (None, "")
            else max(1, _hw_int(cfg, "judge_workers", 4))
        )
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
                "Notebook: edit_workers capped to 1 (use scripts/run_nb2_batch.py for dual-GPU)."
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
    pass

    anomaly_cfg = load_anomaly(
        str(cfg.dataset_name), item.anomaly_id, start=project_root
    )
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
    pass

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
            f"attempt={item.attempt}  method={item.method}  device={stack.device}",
        )
        depth = (
            stack.depth_model.predict(item.source_image)
            if stack.depth_model is not None
            else None
        )
        seg = (
            stack.segmenter.predict(item.source_image)
            if stack.segmenter is not None
            else None
        )
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
                seed_offset=item.seed_offset,
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
            item._edit_error = f"{type(exc).__name__}: {exc}"
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
    pass

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
    return tqdm.write, (tqdm.write if verbose else (lambda _m: None)), True


@dataclass
class _BatchContext:
    """Mutable runtime state shared by batch phase helpers."""

    cfg: Any
    project_root: Path
    synth_dir: Path
    root_dir: Path
    max_retries: int
    target_accepts: dict[str, int] | None
    require_target_boxes: bool
    resume: bool
    dataset: str
    source_hint: str
    base_classes: list[str]
    result: BatchResult
    accepted_counts: dict[str, int]
    done_keys: set[str]
    variation_counters: dict[str, int]
    n_edit: int
    n_judge: int
    methods_in_queue: set[str]
    need_depth: bool
    need_seg: bool
    log: Callable[[str], None]
    log_item: Callable[[str], None]
    use_tqdm: bool
    stacks: list[_EditStack] = field(default_factory=list)
    judge: VLMJudge | None = None
    max_attempts: dict[str, int] | None = None
    seed_pool: dict[str, list[Path]] = field(default_factory=dict)
    method_map: dict[str, str] = field(default_factory=dict)
    seed_cursors: dict[str, int] = field(default_factory=dict)
    oversample: float = 0.1


def _target_met(ctx: _BatchContext, anomaly_id: str) -> bool:
    if not ctx.target_accepts or anomaly_id not in ctx.target_accepts:
        return False
    return ctx.accepted_counts[anomaly_id] >= int(ctx.target_accepts[anomaly_id])


def _attempts_left(ctx: _BatchContext, anomaly_id: str) -> int | None:
    """Remaining edit budget for a class (``None`` = unlimited)."""
    if not ctx.max_attempts or anomaly_id not in ctx.max_attempts:
        return None
    used = ctx.result.stats[anomaly_id].attempts
    return max(0, int(ctx.max_attempts[anomaly_id]) - used)


def _persist_batch_state(ctx: _BatchContext, pending: list[PendingItem]) -> None:
    save_state(
        ctx.root_dir,
        {
            "variation_counters": dict(ctx.variation_counters),
            "seed_cursors": dict(ctx.seed_cursors),
            "stats": {k: asdict(v) for k, v in ctx.result.stats.items()},
            "surplus": dict(ctx.result.surplus),
            "pending": [
                {
                    "anomaly_id": p.anomaly_id,
                    "source_stem": p.source_stem,
                    "source_path": str(p.source_path),
                    "method": p.method,
                    "attempt": int(p.attempt),
                    "pass_index": int(p.pass_index),
                    "variation_index": p.variation_index,
                }
                for p in pending
            ],
        },
    )


def _unload_edit_stacks(ctx: _BatchContext) -> None:
    for stack in ctx.stacks:
        _unload(stack.comparer, stack.annotator, stack.depth_model, stack.segmenter)
    ctx.stacks = []


def _load_edit_stacks(ctx: _BatchContext) -> None:
    _unload(ctx.judge)
    ctx.judge = None
    _unload_edit_stacks(ctx)
    if ctx.n_edit > 1:
        ctx.log(
            f"Edit phase: {ctx.n_edit} process workers (one Klein per GPU). "
            "Note: Flux2Klein list(image)+list(prompt) is multi-ref, not paired batch.",
        )
        return
    ctx.log("Loading edit stack on default device…")
    ctx.stacks.append(
        _load_edit_stack(
            ctx.cfg,
            device=None,
            need_depth=ctx.need_depth,
            need_seg=ctx.need_seg,
            warm_methods=ctx.methods_in_queue,
            log=ctx.log,
        ),
    )


def _load_batch_judge(ctx: _BatchContext) -> None:
    _unload_edit_stacks(ctx)
    if ctx.judge is not None:
        return
    ctx.judge = VLMJudge.from_config(ctx.cfg)
    if ctx.resume and ctx.result.accepted:
        for sample in ctx.result.accepted:
            path = ctx.synth_dir / sample.image_name
            if not path.exists():
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            ctx.judge.register_accepted_embedding(
                sample.anomaly_id, img, path=str(path)
            )
        ctx.log(
            f"Resume: seeded embedding novelty bank with {len(ctx.result.accepted)} accepted image(s)."
        )


def _assign_variations(
    ctx: _BatchContext, items: list[PendingItem]
) -> list[PendingItem]:
    active: list[PendingItem] = []
    admitted: dict[str, int] = {}
    dropped: dict[str, int] = {}
    for item in items:
        if _target_met(ctx, item.anomaly_id):
            continue
        left = _attempts_left(ctx, item.anomaly_id)
        if left is not None and admitted.get(item.anomaly_id, 0) >= left:
            dropped[item.anomaly_id] = dropped.get(item.anomaly_id, 0) + 1
            continue
        admitted[item.anomaly_id] = admitted.get(item.anomaly_id, 0) + 1
        item.variation_index = ctx.variation_counters[item.anomaly_id]
        ctx.variation_counters[item.anomaly_id] = int(item.variation_index) + 1
        active.append(item)
    for anomaly_id, n in dropped.items():
        ctx.log(
            f"  {anomaly_id}: attempt budget exhausted — dropped {n} queued edit(s)."
        )
    return active


def _edit_worker_payloads(
    ctx: _BatchContext, active: list[PendingItem], tmp_root: Path
) -> list[dict[str, Any]]:
    from aieng.syn_data.image.batch.edit_mp import EditJob  # noqa: PLC0415

    for index, item in enumerate(active):
        item._job_id = index  # type: ignore[attr-defined]
    shards = [active[index :: ctx.n_edit] for index in range(ctx.n_edit)]
    payloads: list[dict[str, Any]] = []
    hardware_name = str(
        ctx.cfg.hardware.get("name") or ctx.cfg.get("hardware") or "gpu_l4"
    )
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
                seed_offset=item.seed_offset,
            )
            for item in shard
        ]
        payloads.append(
            {
                "gpu_id": gpu_id,
                "jobs": [job.__dict__ for job in jobs],
                "project_root": str(ctx.project_root),
                "dataset_name": ctx.dataset,
                "hardware": hardware_name,
                "need_depth": ctx.need_depth,
                "need_seg": ctx.need_seg,
                "warm_methods": list(ctx.methods_in_queue),
                "base_classes": ctx.base_classes,
                "tmp_dir": str(tmp_root / f"gpu{gpu_id}"),
            },
        )
    return payloads


def _synthesize_multiprocess(ctx: _BatchContext, active: list[PendingItem]) -> None:
    from aieng.syn_data.image.batch.edit_mp import (  # noqa: PLC0415
        apply_mp_results_to_items,
        mp_synthesize_shard,
    )

    _unload_edit_stacks(ctx)
    tmp_root = Path(tempfile.mkdtemp(prefix="edgecase_edit_"))
    try:
        payloads = _edit_worker_payloads(ctx, active, tmp_root)
        ctx.log(
            f"Dispatching {len(active)} edits across {len(payloads)} GPU processes…"
        )
        mp_ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp_ctx) as pool:
            shard_results = list(pool.map(mp_synthesize_shard, payloads))
        flat = [row for part in shard_results for row in part]
        failed = apply_mp_results_to_items(active, flat)
        # Failed edits count too (same as the single-process path) so the
        # attempt budget always shrinks.
        for item in active:
            ctx.result.stats[item.anomaly_id].attempts += 1
        if failed:
            ctx.log(f"  {len(failed)} edit(s) failed in workers (will retry/reject).")
        ctx.log(f"Process edit done ({len(flat) - len(failed)} ok / {len(flat)} jobs)")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _synthesize_queue(
    ctx: _BatchContext, items: list[PendingItem]
) -> list[PendingItem]:
    """Edit + annotate items within target/budget; return the items actually edited."""
    active = _assign_variations(ctx, items)
    if not active:
        return active
    if ctx.n_edit > 1:
        _synthesize_multiprocess(ctx, active)
        return active
    if not ctx.stacks:
        _load_edit_stacks(ctx)
    assert ctx.stacks
    _synthesize_items(
        active,
        stack=ctx.stacks[0],
        cfg=ctx.cfg,
        project_root=ctx.project_root,
        base_classes=ctx.base_classes,
        target_accepts=ctx.target_accepts,
        accepted_counts=ctx.accepted_counts,
        stats=ctx.result.stats,
        log=ctx.log_item,
        show_progress=ctx.use_tqdm,
    )
    return active


def _judge_api_call(
    ctx: _BatchContext, item: PendingItem
) -> tuple[PendingItem, JudgeResult, bool]:
    assert (
        ctx.judge is not None
        and item.generated is not None
        and item.annotation is not None
    )
    anomaly_cfg = load_anomaly(ctx.dataset, item.anomaly_id, start=ctx.project_root)
    anomaly_classes = list(anomaly_cfg.get("annotation_classes", []) or [])
    targets = target_label_names(item.anomaly_id, anomaly_classes)
    boxed = has_target_detections(item.annotation, targets)
    judgment = ctx.judge.judge(
        item.generated.image,
        prompt=item.generated.prompt,
        anomaly_id=item.anomaly_id,
        anomaly_name=str(anomaly_cfg.get("display_name", item.anomaly_id)),
        annotations_summary=summarize_annotations(item.annotation),
        source_hint=ctx.source_hint,
        require_target_boxes=ctx.require_target_boxes,
        has_target_boxes=boxed,
        exclude_stems={item.source_stem},
        edit_mask=getattr(item.generated, "edit_mask", None),
        annotation=item.annotation,
    )
    return item, judgment, boxed


def _handle_edit_failures(
    ctx: _BatchContext, items: list[PendingItem]
) -> list[PendingItem]:
    retries: list[PendingItem] = []
    for item in items:
        if item.generated is not None and item.annotation is not None:
            continue
        stats = ctx.result.stats[item.anomaly_id]
        err = getattr(item, "_edit_error", None) or "edit produced no image"
        if item.attempt < ctx.max_retries:
            stats.retries += 1
            item.attempt += 1
            item.generated = None
            item.annotation = None
            retries.append(item)
            ctx.log_item(
                f"  edit-fail {item.anomaly_id}  seed={item.source_stem}  → retry ({err})"
            )
        else:
            stats.rejects += 1
            row = {
                "anomaly_id": item.anomaly_id,
                "source_stem": item.source_stem,
                "pass_index": item.pass_index,
                "attempt": item.attempt,
                "decision": "reject",
                "overall": 0.0,
                "has_target_boxes": False,
                "error": err,
            }
            ctx.result.rejected.append(row)
            append_jsonl(rejected_path(ctx.root_dir), row)
            ctx.done_keys.add(item.key)
            ctx.log_item(
                f"  edit-fail {item.anomaly_id}  seed={item.source_stem}  → reject ({err})"
            )
    return retries


def _run_judgments(
    ctx: _BatchContext,
    ready: list[PendingItem],
) -> list[tuple[PendingItem, JudgeResult, bool]]:
    workers = min(ctx.n_judge, len(ready))
    if workers <= 1:
        ready_iter: Any = ready
        if ctx.use_tqdm:
            ready_iter = tqdm(ready, desc="Judging", unit="img", leave=True)
        return [_judge_api_call(ctx, item) for item in ready_iter]
    ctx.log(f"Judging {len(ready)} item(s) with {workers} parallel API calls…")
    judged: list[tuple[PendingItem, JudgeResult, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_api_call, ctx, item) for item in ready]
        done_iter: Any = as_completed(futures)
        if ctx.use_tqdm:
            done_iter = tqdm(
                done_iter, total=len(futures), desc="Judging", unit="img", leave=True
            )
        for future in done_iter:
            judged.append(future.result())
    return judged


def _gate_judgment(
    ctx: _BatchContext,
    item: PendingItem,
    judgment: JudgeResult,
    *,
    boxed: bool,
    anomaly_cfg: Any,
    targets: set[str],
) -> str:
    decision = judgment.decision
    if ctx.require_target_boxes and not boxed:
        decision = "retry" if item.attempt < ctx.max_retries else "reject"
        note = f" [box-gate: no target box → {decision} (open-vocab + edit-mask fallback both empty)]"
        judgment.decision = decision
        judgment.rationale = (judgment.rationale or "").rstrip() + note
    gates = dict(anomaly_cfg.get("accept_gates") or {})
    if decision == "accept" and gates and boxed:
        assert item.generated is not None and item.annotation is not None
        ok, reason = check_box_placement(
            item.annotation, targets, image_size=item.generated.image.size, gates=gates
        )
        if not ok:
            decision = "retry" if item.attempt < ctx.max_retries else "reject"
            note = f" [placement-gate: {reason} → {decision}]"
            judgment.decision = decision
            judgment.rationale = (judgment.rationale or "").rstrip() + note
    return decision


def _log_judgment(
    ctx: _BatchContext,
    item: PendingItem,
    judgment: JudgeResult,
    decision: str,
    boxed: bool,
) -> None:
    ctx.log_item(
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
        + f"  boxes={'yes' if boxed else 'NO'}",
    )


def _accept_judgment(
    ctx: _BatchContext, item: PendingItem, judgment: JudgeResult
) -> None:
    stats = ctx.result.stats[item.anomaly_id]
    stats.accepts += 1
    ctx.accepted_counts[item.anomaly_id] += 1
    pass_tag = f"_p{item.pass_index}" if item.pass_index else ""
    image_name = (
        f"synth_{item.anomaly_id}_{item.source_stem}{pass_tag}_a{item.attempt}.jpg"
    )
    assert (
        item.generated is not None
        and item.annotation is not None
        and ctx.judge is not None
    )
    saved = save_accepted_image(
        item.generated.image, out_dir=ctx.synth_dir, image_name=image_name
    )
    ctx.judge.register_accepted_embedding(
        item.anomaly_id, item.generated.image, path=str(saved)
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
    ctx.result.accepted.append(sample)
    append_jsonl(accepted_path(ctx.root_dir), accepted_sample_to_row(sample))
    ctx.done_keys.add(item.key)


def _reject_judgment(
    ctx: _BatchContext,
    item: PendingItem,
    judgment: JudgeResult,
    decision: str,
    boxed: bool,
) -> None:
    ctx.result.stats[item.anomaly_id].rejects += 1
    row = {
        "anomaly_id": item.anomaly_id,
        "source_stem": item.source_stem,
        "pass_index": item.pass_index,
        "attempt": item.attempt,
        "decision": decision,
        "overall": float(judgment.overall),
        "has_target_boxes": boxed,
    }
    ctx.result.rejected.append(row)
    append_jsonl(rejected_path(ctx.root_dir), row)
    ctx.done_keys.add(item.key)


def _record_judged(
    ctx: _BatchContext,
    item: PendingItem,
    judgment: JudgeResult,
    *,
    outcome: str,
    boxed: bool,
) -> None:
    row = {
        **judge_to_dict(judgment),
        "anomaly_id": item.anomaly_id,
        "source_stem": item.source_stem,
        "pass_index": item.pass_index,
        "attempt": item.attempt,
        "outcome": outcome,
        "has_target_boxes": boxed,
    }
    ctx.result.judged.append(row)
    append_jsonl(judged_path(ctx.root_dir), row)
    if judgment.embedding is not None:
        vec_row = {
            key: row[key]
            for key in ("anomaly_id", "source_stem", "pass_index", "attempt", "outcome")
        }
        vec_row["vector"] = [round(float(v), 5) for v in judgment.embedding]
        ctx.result.embeddings.append(vec_row)
        append_jsonl(embeddings_path(ctx.root_dir), vec_row)


def _apply_judgments(
    ctx: _BatchContext,
    judged: list[tuple[PendingItem, JudgeResult, bool]],
    retries: list[PendingItem],
) -> dict[str, int]:
    counts = {"accept": 0, "retry": 0, "reject": 0, "surplus": 0}
    for item, judgment, boxed in judged:
        anomaly_cfg = load_anomaly(ctx.dataset, item.anomaly_id, start=ctx.project_root)
        anomaly_classes = list(anomaly_cfg.get("annotation_classes", []) or [])
        targets = target_label_names(item.anomaly_id, anomaly_classes)
        decision = _gate_judgment(
            ctx,
            item,
            judgment,
            boxed=boxed,
            anomaly_cfg=anomaly_cfg,
            targets=targets,
        )
        _log_judgment(ctx, item, judgment, decision, boxed)
        if decision == "accept" and _target_met(ctx, item.anomaly_id):
            outcome = "surplus"
            ctx.result.surplus[item.anomaly_id] = (
                ctx.result.surplus.get(item.anomaly_id, 0) + 1
            )
            ctx.done_keys.add(item.key)
        elif decision == "accept":
            outcome = "accept"
            _accept_judgment(ctx, item, judgment)
        elif decision == "retry" and item.attempt < ctx.max_retries:
            outcome = "retry"
            ctx.result.stats[item.anomaly_id].retries += 1
        else:
            outcome = "reject"
            _reject_judgment(ctx, item, judgment, decision, boxed)
        counts[outcome] += 1
        _record_judged(ctx, item, judgment, outcome=outcome, boxed=boxed)
        if outcome == "retry":
            item.attempt += 1
            item.generated = None
            item.annotation = None
            retries.append(item)
    return counts


def _judge_queue(ctx: _BatchContext, items: list[PendingItem]) -> list[PendingItem]:
    """Judge items and return those that should retry."""
    assert ctx.judge is not None
    retries = _handle_edit_failures(ctx, items)
    ready = [
        item
        for item in items
        if item.generated is not None
        and item.annotation is not None
        and not _target_met(ctx, item.anomaly_id)
    ]
    if not ready:
        return retries
    counts = _apply_judgments(ctx, _run_judgments(ctx, ready), retries)
    if not ctx.result.real_embeddings:
        banks = ctx.judge.real_embedding_bank(list(ctx.seed_pool))
        if banks:
            ctx.result.real_embeddings = banks
            save_real_embeddings(ctx.root_dir, banks)
    surplus = f"  surplus={counts['surplus']}" if counts["surplus"] else ""
    ctx.log(
        f"Judge round: accept={counts['accept']}  retry={counts['retry']}  "
        f"reject={counts['reject']}{surplus}  accepted_total={len(ctx.result.accepted)}",
    )
    return retries


def _draw_seeds(
    ctx: _BatchContext, anomaly_id: str, n: int, claimed: set[str]
) -> list[PendingItem]:
    """Take the next ``n`` seeds: unused scenes first, then revisits (pass ≥ 1)."""
    pool = ctx.seed_pool.get(anomaly_id) or []
    out: list[PendingItem] = []
    tries = 0
    while pool and len(out) < n and tries < n + len(pool):
        tries += 1
        cursor = ctx.seed_cursors.get(anomaly_id, 0)
        ctx.seed_cursors[anomaly_id] = cursor + 1
        path = pool[cursor % len(pool)]
        item = PendingItem(
            anomaly_id=anomaly_id,
            method=ctx.method_map[anomaly_id],
            source_path=path,
            source_image=Image.open(path).convert("RGB"),
            source_stem=path.stem,
            pass_index=cursor // len(pool),
        )
        if item.key in ctx.done_keys or item.key in claimed:
            continue
        claimed.add(item.key)
        out.append(item)
    return out


def _refill_items(ctx: _BatchContext, pending: list[PendingItem]) -> list[PendingItem]:
    """Queue new seeds for classes still below target (budget permitting).

    Round size ≈ ``(1 + oversample) × deficit / acceptance rate``, filled first
    by the retries already in flight and then by new seeds (rate = 1 before
    anything has been judged, so the first round is ``target × (1 + oversample)``).
    New seeds are unused scenes while the pool lasts, then revisits. The
    per-class attempt budget bounds the total, so the loop always terminates.
    """
    if not ctx.target_accepts or not ctx.max_attempts:
        return []
    claimed = {p.key for p in pending}
    new_items: list[PendingItem] = []
    for anomaly_id, pool in ctx.seed_pool.items():
        left = _attempts_left(ctx, anomaly_id)
        if not pool or left is None or _target_met(ctx, anomaly_id):
            continue
        in_flight = sum(1 for p in pending if p.anomaly_id == anomaly_id)
        left -= in_flight
        if left <= 0:
            continue
        st = ctx.result.stats[anomaly_id]
        rate = st.accepts / st.attempts if st.attempts else 1.0
        deficit = int(ctx.target_accepts[anomaly_id]) - ctx.accepted_counts[anomaly_id]
        want = (1.0 + ctx.oversample) * deficit - in_flight * rate
        n_new = math.ceil(round(max(0.0, want) / max(rate, _MIN_REFILL_RATE), 6))
        n_new = min(n_new, left, len(pool))
        new_items.extend(_draw_seeds(ctx, anomaly_id, n_new, claimed))
    if new_items:
        summary = {
            aid: {
                "new_scenes": sum(
                    1 for i in new_items if i.anomaly_id == aid and i.pass_index == 0
                ),
                "reused_scenes": sum(
                    1 for i in new_items if i.anomaly_id == aid and i.pass_index > 0
                ),
            }
            for aid in ctx.seed_pool
        }
        ctx.log(f"Queue new seeds (below target): {summary}")
    return new_items


def _pending_from_state(
    ctx: _BatchContext, rows: list[dict[str, Any]]
) -> list[PendingItem]:
    """Rebuild in-flight items saved in ``state.json`` (resume)."""
    items: list[PendingItem] = []
    seen: set[str] = set()
    for row in rows:
        anomaly_id = str(row.get("anomaly_id") or "")
        path = Path(str(row.get("source_path") or ""))
        if anomaly_id not in ctx.seed_pool or not path.exists():
            continue
        item = PendingItem(
            anomaly_id=anomaly_id,
            method=str(row.get("method") or ctx.method_map[anomaly_id]),
            source_path=path,
            source_image=Image.open(path).convert("RGB"),
            source_stem=path.stem,
            attempt=int(row.get("attempt") or 0),
            pass_index=int(row.get("pass_index") or 0),
        )
        if (
            item.key in ctx.done_keys
            or item.key in seen
            or _target_met(ctx, anomaly_id)
        ):
            continue
        seen.add(item.key)
        items.append(item)
    return items


def _initial_queue(
    ctx: _BatchContext, saved_pending: list[dict[str, Any]]
) -> list[PendingItem]:
    """First edit round (fresh run or resume)."""
    if ctx.target_accepts and ctx.max_attempts:
        pending = _pending_from_state(ctx, saved_pending) if ctx.resume else []
        return pending + _refill_items(ctx, pending)
    # No attempt budget: one pass over every seed (each tried ≤ 1 + max_retries times).
    queue: list[PendingItem] = []
    for anomaly_id, pool in ctx.seed_pool.items():
        if _target_met(ctx, anomaly_id):
            continue
        ctx.seed_cursors[anomaly_id] = 0
        queue.extend(_draw_seeds(ctx, anomaly_id, len(pool), set()))
    return queue


def _restore_batch_checkpoint(
    root_dir: Path,
    result: BatchResult,
    *,
    accepted_counts: dict[str, int],
    variation_counters: dict[str, int],
    seed_cursors: dict[str, int],
    anomaly_ids: list[str],
    log: Callable[[str], None],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Restore accepts/rejects/stats/counters; return done keys + in-flight rows."""
    checkpoint = load_checkpoint(root_dir)
    result.accepted = list(checkpoint["accepted"])
    result.rejected = list(checkpoint["rejected"])
    result.judged = list(checkpoint.get("judged") or [])
    result.real_embeddings, result.embeddings = load_embedding_artifacts(root_dir)
    done_keys = set(checkpoint["accepted_keys"]) | set(checkpoint["rejected_keys"])
    for sample in result.accepted:
        accepted_counts[sample.anomaly_id] = (
            accepted_counts.get(sample.anomaly_id, 0) + 1
        )
    state = checkpoint.get("state") or {}
    for anomaly_id, value in (state.get("variation_counters") or {}).items():
        variation_counters[str(anomaly_id)] = int(value)
    for anomaly_id, value in (state.get("seed_cursors") or {}).items():
        seed_cursors[str(anomaly_id)] = int(value)
    for anomaly_id, value in (state.get("surplus") or {}).items():
        result.surplus[str(anomaly_id)] = int(value)
    result.stats = rebuild_stats(
        anomaly_ids,
        accepted=result.accepted,
        rejected=result.rejected,
        state=checkpoint.get("state"),
    )
    log(
        f"Resume: loaded {len(result.accepted)} accepted, "
        f"{len(result.rejected)} rejected; skipping {len(done_keys)} seeds.",
    )
    return done_keys, list(state.get("pending") or [])


def _prepare_batch_context(
    seeds_by_anomaly: dict[str, list[Path]],
    method_map: dict[str, str],
    *,
    cfg: Any,
    project_root: Path,
    synth_dir: Path,
    max_retries: int,
    target_accepts: dict[str, int] | None,
    max_attempts: dict[str, int] | None,
    oversample: float,
    require_target_boxes: bool,
    progress: Any,
    verbose: bool,
    resume: bool,
    nb2_dir: Path | str | None,
) -> tuple[_BatchContext, list[PendingItem], str | None]:
    """Initialize mutable state and model requirements for a batch run."""
    log, log_item, use_tqdm = _resolve_progress(progress, verbose=verbose)
    previous_progress = os.environ.get("EDGECASE_DISABLE_PIPE_PROGRESS")
    if verbose and not use_tqdm:
        os.environ.pop("EDGECASE_DISABLE_PIPE_PROGRESS", None)
    else:
        os.environ["EDGECASE_DISABLE_PIPE_PROGRESS"] = "1"
    synth_dir = Path(synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(nb2_dir) if nb2_dir is not None else synth_dir.parent
    root_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult(
        stats={aid: ClassRunStats(anomaly_id=aid) for aid in seeds_by_anomaly},
        targets=dict(target_accepts or {}),
        max_attempts=dict(max_attempts or {}),
    )
    accepted_counts = dict.fromkeys(seeds_by_anomaly, 0)
    variation_counters: dict[str, int] = dict.fromkeys(seeds_by_anomaly, 0)
    seed_cursors: dict[str, int] = {}
    done_keys: set[str] = set()
    saved_pending: list[dict[str, Any]] = []
    if resume:
        done_keys, saved_pending = _restore_batch_checkpoint(
            root_dir,
            result,
            accepted_counts=accepted_counts,
            variation_counters=variation_counters,
            seed_cursors=seed_cursors,
            anomaly_ids=list(seeds_by_anomaly),
            log=log,
        )
    open_classes = [
        aid
        for aid in seeds_by_anomaly
        if not target_accepts
        or aid not in target_accepts
        or accepted_counts.get(aid, 0) < int(target_accepts[aid])
    ]
    methods = {method_map[aid] for aid in open_classes}
    need_depth, need_seg = _methods_need_conditioning(methods)
    n_edit, n_judge = 1, 1
    if open_classes:
        n_edit = _edit_workers(cfg, log=log)
        n_judge = _judge_workers(cfg)
        instruct_id = str(cfg.generation.get("instruct_model_id") or "")
        inpaint_id = str(cfg.generation.get("inpaint_model_id") or "")
        if ("instruct" in methods and MethodComparer._is_klein_model(instruct_id)) or (
            "inpaint" in methods and MethodComparer._is_klein_model(inpaint_id)
        ):
            assert_klein_available()
    context = _BatchContext(
        cfg=cfg,
        project_root=project_root,
        synth_dir=synth_dir,
        root_dir=root_dir,
        max_retries=max_retries,
        target_accepts=target_accepts,
        require_target_boxes=require_target_boxes,
        resume=resume,
        dataset=str(cfg.dataset_name),
        source_hint=str(cfg.dataset.get("source_hint", "a real photograph")),
        base_classes=list(cfg.annotation.classes),
        result=result,
        accepted_counts=accepted_counts,
        done_keys=done_keys,
        variation_counters=variation_counters,
        n_edit=n_edit,
        n_judge=n_judge,
        methods_in_queue=methods,
        need_depth=need_depth,
        need_seg=need_seg,
        log=log,
        log_item=log_item,
        use_tqdm=use_tqdm,
        max_attempts=dict(max_attempts) if max_attempts else None,
        seed_pool={aid: list(paths) for aid, paths in seeds_by_anomaly.items()},
        method_map=dict(method_map),
        seed_cursors=seed_cursors,
        oversample=max(0.0, float(oversample)),
    )
    queue = _initial_queue(context, saved_pending)
    if queue:
        context.log(
            f"Batch parallelism: edit_workers={context.n_edit}  judge_workers={context.n_judge}  "
            f"depth={need_depth}  seg={need_seg}",
        )
    return context, queue, previous_progress


def run_batch_synthesis(
    seeds_by_anomaly: dict[str, list[Path]],
    method_map: dict[str, str],
    *,
    cfg: Any,
    project_root: Path,
    synth_dir: Path,
    max_retries: int = 2,
    target_accepts: dict[str, int] | None = None,
    max_attempts: dict[str, int] | int | None = None,
    oversample: float = 0.1,
    require_target_boxes: bool = True,
    progress: Any = "tqdm",
    verbose: bool = False,
    resume: bool = False,
    nb2_dir: Path | str | None = None,
) -> BatchResult:
    r"""Generate, annotate, and judge many seeds with models loaded once per phase.

    Phases:
      1. Load edit stack(s) → synthesize + annotate (optionally 1 stack per GPU)
      2. Unload edit stack → load judge → concurrent API decisions
      3. On retries: reload edit stack(s), re-edit, re-judge
      4. Classes still below ``target_accepts`` draw more seeds (only when
         ``max_attempts`` is set)

    ``target_accepts`` is the number of *accepted* images wanted per class; a
    class stops as soon as it gets there. ``max_attempts`` (int = same for every
    class) caps edits per class, counting first edits, retries, and revisits.

    With a budget, ``seeds_by_anomaly`` is a *pool*: the first round edits
    ``target × (1 + oversample)`` unused scenes; each later round is sized
    ``(1 + oversample) × deficit / acceptance rate``, filled by pending retries
    and then new seeds — unused scenes while they last, then revisits of used
    scenes with a new variation + diffusion seed. When the budget runs out
    first the run ends normally and ``BatchResult.target_reached()`` reports
    which classes fell short. With ``max_attempts=None`` every seed in the pool
    is edited once, with up to ``max_retries`` retries.

    When ``require_target_boxes`` is True (default), an item cannot be accepted
    without at least one target-class box (YOLO-World or edit-mask fallback).

    Set ``resume=True`` to skip ``(anomaly, seed)`` pairs already accepted or
    finally rejected under ``<nb2_dir>/checkpoint/``.

    ``progress``: ``\"tqdm\"`` (default), ``\"print\"``, ``\"silent\"``, or a callable.
    Per-item lines are off unless ``verbose=True``.
    """
    started = time.perf_counter()
    if isinstance(max_attempts, int):
        max_attempts = dict.fromkeys(seeds_by_anomaly, max_attempts)
    ctx, queue, previous_progress = _prepare_batch_context(
        seeds_by_anomaly,
        method_map,
        cfg=cfg,
        project_root=project_root,
        synth_dir=synth_dir,
        max_retries=max_retries,
        target_accepts=target_accepts,
        max_attempts=max_attempts,
        oversample=oversample,
        require_target_boxes=require_target_boxes,
        progress=progress,
        verbose=verbose,
        resume=resume,
        nb2_dir=nb2_dir,
    )
    log = ctx.log
    result = ctx.result

    active = queue
    if active:
        log(f"Loading edit stack once ({len(active)} jobs)…")
        _load_edit_stacks(ctx)
        active = _synthesize_queue(ctx, active)
        _persist_batch_state(ctx, active)
    else:
        log("No seeds queued — nothing to synthesize (targets met or all done).")

    while active:
        log("Switching to judge…")
        _load_batch_judge(ctx)
        retries = _judge_queue(ctx, active)
        retries = [r for r in retries if not _target_met(ctx, r.anomaly_id)]
        retries += _refill_items(ctx, retries)
        _persist_batch_state(ctx, retries)
        if not retries:
            break
        log(f"Re-editing {len(retries)} item(s)…")
        _load_edit_stacks(ctx)
        active = _synthesize_queue(ctx, retries)
        _persist_batch_state(ctx, active)

    _unload_edit_stacks(ctx)
    _unload(ctx.judge)
    if previous_progress is None:
        os.environ.pop("EDGECASE_DISABLE_PIPE_PROGRESS", None)
    else:
        os.environ["EDGECASE_DISABLE_PIPE_PROGRESS"] = previous_progress
    _persist_batch_state(ctx, [])
    result.elapsed_s = time.perf_counter() - started
    for aid, pool in ctx.seed_pool.items():
        drawn = ctx.seed_cursors.get(aid, 0)
        result.scene_usage[aid] = {
            "pool": len(pool),
            "new": min(drawn, len(pool)),
            "reused": max(0, drawn - len(pool)),
        }
    log(
        f"Done. Accepted {len(result.accepted)} / {sum(s.attempts for s in result.stats.values())} attempts."
    )
    for aid, reached in result.target_reached().items():
        if not reached:
            log(
                f"  {aid}: target NOT reached ({result.accepted_counts().get(aid, 0)}"
                f"/{result.targets[aid]} accepted; attempt budget or seed pool exhausted)."
            )
    return result


def batch_summary_rows(result: BatchResult) -> list[dict[str, Any]]:
    """Per-class run summary (target, accepted, attempts, retries, rejections, rate)."""
    counts = result.accepted_counts()
    rows: list[dict[str, Any]] = []
    for aid, st in result.stats.items():
        target = result.targets.get(aid)
        rows.append(
            {
                "anomaly_id": aid,
                "target": target,
                "accepted": counts.get(aid, 0),
                "target_reached": None
                if target is None
                else counts.get(aid, 0) >= target,
                "attempts": st.attempts,
                "max_attempts": result.max_attempts.get(aid),
                "retries": st.retries,
                "rejects": st.rejects,
                "surplus": result.surplus.get(aid, 0),
                "acceptance_rate": st.acceptance_rate,
                **{
                    f"scenes_{k}": v
                    for k, v in (result.scene_usage.get(aid) or {}).items()
                },
            },
        )
    return rows


def format_batch_summary(result: BatchResult) -> str:
    """Human-readable batch summary; flags classes that missed their target."""
    lines = [
        f"{'class':16s} {'target':>6s} {'accepted':>8s} {'attempts':>13s} "
        f"{'retries':>7s} {'rejects':>7s} {'accept rate':>11s} {'scenes new/reused/pool':>22s}  status",
    ]
    for row in batch_summary_rows(result):
        scenes = (
            f"{row['scenes_new']}/{row['scenes_reused']}/{row['scenes_pool']}"
            if "scenes_pool" in row
            else "-"
        )
        budget = f"/{row['max_attempts']}" if row["max_attempts"] is not None else ""
        target = "-" if row["target"] is None else str(row["target"])
        if row["target_reached"] is None:
            status = ""
        elif row["target_reached"]:
            status = "target reached"
        else:
            status = "TARGET NOT REACHED"
        if row["surplus"]:
            status += f"  (+{row['surplus']} surplus passes not exported)"
        lines.append(
            f"{row['anomaly_id']:16s} {target:>6s} {row['accepted']:>8d} "
            f"{str(row['attempts']) + budget:>13s} {row['retries']:>7d} {row['rejects']:>7d} "
            f"{100 * row['acceptance_rate']:>10.1f}% {scenes:>22s}  {status}",
        )
    total_acc = len(result.accepted)
    total_att = sum(s.attempts for s in result.stats.values())
    lines.append(
        f"Total: {total_acc} accepted / {total_att} edit attempts"
        + (f" ({100 * total_acc / total_att:.1f}%)" if total_att else "")
        + (f"  ·  elapsed {result.elapsed_s / 60:.1f} min" if result.elapsed_s else ""),
    )
    return "\n".join(lines)
