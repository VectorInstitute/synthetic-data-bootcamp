#!/usr/bin/env python3
r"""CLI for Notebook 2 batch synthesis (background / dual-GPU friendly).

Examples::

  # Full dual-GPU run on 2×L4
  python scripts/run_nb2_batch.py --hardware gpu_l4x2

  # Resume after a kill / disconnect
  python scripts/run_nb2_batch.py --hardware gpu_l4x2 --resume

  # Smoke test
  python scripts/run_nb2_batch.py --hardware gpu_l4 \\
      --n-synth-seeds traffic_cone=4,trash_bin=4 \\
      --target-accepted traffic_cone=2,trash_bin=2 --max-attempts 12

Run from the implementation root. Prefer this over the NB2 notebook cell for
``parallel_edit_workers>1`` (Jupyter + ProcessPool + CUDA is flaky).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from aieng.syn_data.image.batch.checkpoint import write_split_snapshot
from aieng.syn_data.image.batch.export import export_nb2_dataset
from aieng.syn_data.image.batch.runner import format_batch_summary, run_batch_synthesis
from aieng.syn_data.image.bootstrap import bootstrap_project_root
from aieng.syn_data.image.config import load_config, load_env
from aieng.syn_data.image.data import prepare_sample_images
from aieng.syn_data.image.data.eda import (
    allocate_budget,
    clamp_counts,
    group_by_tag,
    list_tagged_images,
    load_labels_for_dir,
    pick_synth_seeds,
    stratified_holdout,
    summarize_distribution,
    write_json,
)
from aieng.syn_data.image.generate.pipeline import resolve_method_map


def _parse_kv_ints(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for raw_part in raw.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected key=int, got {part!r}")
        key, val = part.split("=", 1)
        out[key.strip()] = int(val.strip())
    return out


def _parse_methods(raw: list[str] | None, workshop: list[str]) -> dict[str, str]:
    if not raw:
        return dict.fromkeys(workshop, "instruct")
    out: dict[str, str] = {}
    for part in raw:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected anomaly=method, got {part!r}")
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    for aid in workshop:
        out.setdefault(aid, "instruct")
    return out


def _build_parser() -> argparse.ArgumentParser:
    """Build the Notebook 2 batch CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mapillary_vistas")
    parser.add_argument("--hardware", default="gpu_l4x2")
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="anomaly=method (repeatable). Default: instruct for all workshop anomalies.",
    )
    parser.add_argument(
        "--test-counts",
        default="scene=90,traffic_cone=30,trash_bin=30",
        help="tag=count,... for real holdout",
    )
    parser.add_argument(
        "--n-synth-seeds",
        default="traffic_cone=300,trash_bin=300",
        help="anomaly=count,... scene pool per class (unused scenes are drawn first)",
    )
    parser.add_argument(
        "--target-accepted",
        "--target-accepts",
        dest="target_accepted",
        default="traffic_cone=100,trash_bin=100",
        help="anomaly=count,... accepted images wanted per class (a class stops when hit)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=400,
        help="Per-class edit budget (first edits + retries + seed revisits); 0 = no revisits",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=0.1,
        help="Extra edits per round on top of the deficit/rate estimate (0.1 = 10%%)",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image edit/judge lines (default: tqdm + phase summaries only).",
    )
    parser.add_argument(
        "--progress",
        choices=("tqdm", "print", "silent"),
        default="tqdm",
        help="Progress style for the batch loop (default: tqdm).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip final manifest export (checkpoint images still written)",
    )
    parser.add_argument(
        "--require-target-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _prepare_run(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    """Prepare source splits, synthesis seeds, and checkpoint metadata."""
    load_env(project_root)
    cfg = load_config(
        start=project_root,
        overrides=[f"dataset_name={args.dataset}", f"hardware={args.hardware}"],
    )
    prepare_sample_images(cfg=cfg)

    workshop = list(cfg.dataset.workshop_anomalies)
    method_by_anomaly = _parse_methods(args.method, workshop)
    method_map = resolve_method_map(method_by_anomaly, workshop, cfg=cfg)
    test_counts_req = _parse_kv_ints(args.test_counts)
    n_synth_req = _parse_kv_ints(args.n_synth_seeds)
    target_accepted_cfg = _parse_kv_ints(args.target_accepted)
    focus_tags = ["scene", *workshop]
    stem_prefixes = list(cfg.data.get("stem_prefixes") or focus_tags)

    samples_dir = Path(cfg.paths.samples_dir)
    all_paths = list_tagged_images(samples_dir)
    by_tag = group_by_tag(all_paths, tags=focus_tags, prefixes=stem_prefixes)
    labels = load_labels_for_dir(samples_dir)

    available = {t: len(by_tag.get(t) or []) for t in focus_tags}
    test_counts = clamp_counts(test_counts_req, available, min_remaining=1)
    train_real, test = stratified_holdout(by_tag, test_counts, seed=args.split_seed)

    # Notebook uses outputs/<dataset>/nb2
    output_dir = Path(cfg.paths.outputs_dir) / "nb2"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "split_summary.json",
        {
            "test_counts": test_counts,
            "train_real": summarize_distribution(train_real, focus_tags=focus_tags),
            "test": summarize_distribution(test, focus_tags=focus_tags),
            "seed": args.split_seed,
        },
    )

    scene_train = list(train_real.get("scene") or [])
    seed_request = allocate_budget(n_synth_req, len(scene_train))
    seeds_by_anomaly = pick_synth_seeds(
        scene_train,
        seed_request,
        labels=labels,
        rare_classes=workshop,
        seed=args.split_seed,
    )
    target_accepted = {k: int(target_accepted_cfg.get(k, len(v))) for k, v in seeds_by_anomaly.items()}
    max_attempts = dict.fromkeys(seeds_by_anomaly, args.max_attempts) if args.max_attempts > 0 else None

    config_snapshot = {
        "dataset": args.dataset,
        "hardware": args.hardware,
        "method_by_anomaly": dict(method_map),
        "test_counts": test_counts,
        "n_synth_seeds": seed_request,
        "target_accepted": target_accepted,
        "max_attempts": max_attempts,
        "oversample": args.oversample,
        "max_retries": args.max_retries,
        "split_seed": args.split_seed,
        "judge_model": str(cfg.judge.model_id),
        "judge_threshold": float(cfg.judge.threshold),
        "generation_family": str(cfg.generation.family),
        "resume": bool(args.resume),
    }
    write_split_snapshot(
        output_dir,
        train_real=train_real,
        test=test,
        seeds_by_anomaly=seeds_by_anomaly,
        method_map=method_map,
        target_accepts=target_accepted,
        config_snapshot=config_snapshot,
    )
    return {
        "cfg": cfg,
        "output_dir": output_dir,
        "train_real": train_real,
        "test": test,
        "labels": labels,
        "seeds_by_anomaly": seeds_by_anomaly,
        "method_map": method_map,
        "target_accepted": target_accepted,
        "max_attempts": max_attempts,
        "config_snapshot": config_snapshot,
    }


def _print_run_summary(args: argparse.Namespace, project_root: Path, run: dict[str, Any]) -> None:
    """Print the resolved batch configuration."""
    print("NB2 CLI")
    print(f"  project_root = {project_root}")
    print(f"  output_dir   = {run['output_dir']}")
    print(f"  hardware     = {args.hardware}")
    print(f"  methods      = {run['method_map']}")
    print(f"  seeds        = { {k: len(v) for k, v in run['seeds_by_anomaly'].items()} }")
    print(f"  targets      = {run['target_accepted']}  (accepted images per class)")
    print(f"  max_attempts = {run['max_attempts']}")
    print(f"  resume       = {args.resume}")


def _run_batch(args: argparse.Namespace, project_root: Path, run: dict[str, Any]) -> Any:
    """Execute synthesis using the prepared run inputs."""
    output_dir = run["output_dir"]
    return run_batch_synthesis(
        run["seeds_by_anomaly"],
        run["method_map"],
        cfg=run["cfg"],
        project_root=project_root,
        synth_dir=output_dir / "synthetic",
        max_retries=args.max_retries,
        target_accepts=run["target_accepted"],
        max_attempts=run["max_attempts"],
        oversample=args.oversample,
        require_target_boxes=args.require_target_boxes,
        resume=args.resume,
        nb2_dir=output_dir,
        progress=args.progress,
        verbose=bool(args.verbose),
    )


def _report_and_export(args: argparse.Namespace, run: dict[str, Any], batch: Any) -> int:
    """Print acceptance statistics and optionally export the dataset."""
    print("\nBatch summary:")
    print(format_batch_summary(batch))
    run["config_snapshot"]["target_reached"] = batch.target_reached()
    run["config_snapshot"]["scene_usage"] = batch.scene_usage

    if args.no_export:
        print("Skipped export (--no-export). Checkpoint under", run["output_dir"] / "checkpoint")
        return 0

    paths = export_nb2_dataset(
        output_dir=run["output_dir"],
        accepted=batch.accepted,
        train_real=run["train_real"],
        test=run["test"],
        real_labels=run["labels"],
        run_stats=batch.stats,
        config_snapshot=run["config_snapshot"],
    )
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the batch generation command."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = _build_parser().parse_args(argv)
    project_root = bootstrap_project_root(Path(__file__).resolve().parent.parent)
    run = _prepare_run(args, project_root)
    _print_run_summary(args, project_root, run)
    batch = _run_batch(args, project_root, run)
    return _report_and_export(args, run, batch)


if __name__ == "__main__":
    raise SystemExit(main())
