"""Visualization helpers for notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from aieng.syn_data.image.data.loader import ImageSample
from aieng.syn_data.image.generate.annotation import AnnotationResult
from aieng.syn_data.image.generate.compare_methods import COMPARE_METHODS, METHOD_SPECS
from aieng.syn_data.image.generate.conditioning import DepthResult, SegmentationResult
from aieng.syn_data.image.generate.generation import GenerationResult


def show_image(
    image: Image.Image | np.ndarray, *, title: str = "", ax: Axes | None = None
) -> Axes:
    """Display a single image on a matplotlib axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if isinstance(image, Image.Image):
        ax.imshow(image)
    else:
        ax.imshow(image.astype(np.uint8) if image.dtype != np.float32 else image)
    ax.set_title(title)
    ax.axis("off")
    return ax


def show_samples(
    samples: list[ImageSample],
    *,
    ncol: int = 3,
    figsize: tuple[float, float] = (12, 4),
) -> tuple[Figure, Any]:
    """Grid of loaded real images."""
    n = len(samples)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, sample in zip(axes, samples):
        show_image(sample.image, title=sample.name, ax=ax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Real source images", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, axes


def show_depth_result(
    sample: ImageSample,
    depth: DepthResult,
    *,
    figsize: tuple[float, float] = (12, 4),
) -> tuple[Figure, Any]:
    """Original | grayscale depth | colormap depth."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    show_image(sample.image, title="Original", ax=axes[0])
    axes[1].imshow(depth.depth_map, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Depth map (normalized)")
    axes[1].axis("off")
    show_image(depth.colormap, title="Depth colormap", ax=axes[2])

    fig.suptitle(f"Depth estimation — {sample.name}", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, axes


def show_segmentation_result(
    sample: ImageSample,
    seg: SegmentationResult,
    *,
    figsize: tuple[float, float] = (12, 4),
) -> tuple[Figure, Any]:
    """Original | colored regions | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    show_image(sample.image, title="Original", ax=axes[0])
    show_image(seg.colored_map, title=f"Regions ({seg.num_regions})", ax=axes[1])
    show_image(seg.overlay, title="Overlay", ax=axes[2])

    fig.suptitle(f"Segmentation — {sample.name}", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, axes


def show_structure_overview(
    sample: ImageSample,
    depth: DepthResult,
    seg: SegmentationResult,
    *,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[Figure, Any]:
    """2×2 panel: original, depth, segmentation, overlay — for notebook summaries."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    show_image(sample.image, title="1. Real image", ax=axes[0, 0])
    show_image(
        depth.colormap, title="2. Depth (ControlNet conditioning)", ax=axes[0, 1]
    )
    show_image(seg.colored_map, title="3. Segmentation map", ax=axes[1, 0])
    show_image(seg.overlay, title="4. Segmentation overlay", ax=axes[1, 1])

    fig.suptitle(
        f"Structure extraction — {sample.name}\n(these maps guide diffusion; they are NOT sent to the VLM judge later)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig, axes


def save_structure_artifacts(
    sample: ImageSample,
    depth: DepthResult,
    seg: SegmentationResult,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist conditioning maps for the next pipeline stage."""
    root = Path(output_dir)
    depth_dir = root / "depth"
    seg_dir = root / "segmentation"
    depth_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "depth_npy": depth_dir / f"{sample.name}_depth.npy",
        "depth_png": depth_dir / f"{sample.name}_depth.png",
        "segmentation_png": seg_dir / f"{sample.name}_seg.png",
        "overlay_png": seg_dir / f"{sample.name}_overlay.png",
    }

    np.save(paths["depth_npy"], depth.depth_map)
    Image.fromarray(depth.colormap).save(paths["depth_png"])
    Image.fromarray(seg.colored_map).save(paths["segmentation_png"])
    Image.fromarray(seg.overlay).save(paths["overlay_png"])

    if seg.edit_mask is not None:
        mask_path = seg_dir / f"{sample.name}_edit_mask.png"
        Image.fromarray((seg.edit_mask.astype(np.uint8) * 255)).save(mask_path)
        paths["edit_mask_png"] = mask_path

    return paths


def show_generation_result(
    sample: ImageSample,
    generated: GenerationResult,
    *,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Any]:
    """Original vs ControlNet edit with the prompt used."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    show_image(sample.image, title="Source", ax=axes[0])
    title = "Synthetic edit"
    if generated.anomaly_id:
        method = generated.method or ""
        title = f"{generated.anomaly_id}" + (f" [{method}]" if method else "")
    show_image(generated.image, title=title, ax=axes[1])
    prompt_preview = (
        generated.prompt
        if len(generated.prompt) < 140
        else generated.prompt[:137] + "..."
    )
    fig.suptitle(
        f"Anomaly edit — {sample.name}\nPrompt: {prompt_preview}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig, axes


def show_method_comparison(
    sample: ImageSample,
    bundle: Any,
    *,
    methods: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """Notebook 0.5 panel: original | mask/depth/seg | method outputs."""
    pass

    methods = tuple(methods) if methods is not None else tuple(COMPARE_METHODS)
    n_out = max(len(methods), 1)
    ncols = max(4, n_out)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.15], hspace=0.25, wspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    show_image(sample.image, title="Original", ax=ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    if bundle.edit_mask is not None:
        show_image(
            (bundle.edit_mask.astype(np.uint8) * 255),
            title="Inpaint mask",
            ax=ax1,
        )
    else:
        ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    show_image(bundle.depth.colormap, title="Depth", ax=ax2)

    ax3 = fig.add_subplot(gs[0, 3])
    show_image(bundle.segmentation.colored_map, title="Segmentation", ax=ax3)
    for j in range(4, ncols):
        fig.add_subplot(gs[0, j]).axis("off")

    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, i])
        result = bundle.results.get(method)
        spec = METHOD_SPECS.get(method)
        title = spec.title if spec else method
        if result is None:
            ax.set_title(title)
            ax.axis("off")
            continue
        if getattr(result, "error", None):
            title = f"{title}\n(skipped)"
        show_image(result.image, title=title, ax=ax)

    for j in range(len(methods), ncols):
        ax = fig.add_subplot(gs[1, j])
        ax.axis("off")
        if j == len(methods):
            ax.text(
                0.05,
                0.5,
                "\n".join(
                    f"• {METHOD_SPECS[m].title}: {METHOD_SPECS[m].summary}"
                    for m in methods
                    if m in METHOD_SPECS
                ),
                va="center",
                fontsize=9,
                wrap=True,
                transform=ax.transAxes,
            )

    fig.suptitle(
        f"Method comparison — {sample.name} / {bundle.anomaly_id}\n{bundle.prompt[:120]}",
        fontsize=12,
        y=0.98,
    )
    return fig


def save_compare_artifacts(
    sample: ImageSample,
    bundle: Any,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist Notebook 0.5 compare outputs under outputs/compare/."""
    root = Path(output_dir) / "compare" / f"{sample.name}_{bundle.anomaly_id}"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"dir": root}
    sample.image.save(root / "original.jpg")
    Image.fromarray(bundle.depth.colormap).save(root / "depth.png")
    Image.fromarray(bundle.segmentation.colored_map).save(root / "seg.png")
    if bundle.edit_mask is not None:
        Image.fromarray((bundle.edit_mask.astype(np.uint8) * 255)).save(
            root / "mask.png"
        )
    meta = {
        "sample": sample.name,
        "anomaly_id": bundle.anomaly_id,
        "prompt": bundle.prompt,
        "methods": {},
    }
    for method, result in bundle.results.items():
        out = root / f"{method}.png"
        result.image.save(out)
        paths[method] = out
        meta["methods"][method] = {
            "prompt_used": result.prompt,
            "seed": result.seed,
        }
    meta_path = root / "meta.json"
    pass

    meta_path.write_text(json.dumps(meta, indent=2))
    paths["meta"] = meta_path
    return paths


def show_annotation_result(
    image: Image.Image | np.ndarray,
    annotation: AnnotationResult,
    *,
    title: str = "Annotations",
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Any]:
    """Original vs open-vocabulary detections (YOLO-World boxes)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    show_image(image, title="Image", ax=axes[0])
    show_image(
        annotation.overlay,
        title=f"Detections ({annotation.num_instances})",
        ax=axes[1],
    )
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig, axes


def save_generation_artifact(
    sample: ImageSample,
    generated: GenerationResult,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist a synthetic edit, prompt metadata, and optional edit mask."""
    root = Path(output_dir) / "generated"
    root.mkdir(parents=True, exist_ok=True)

    suffix = generated.anomaly_id or "synthetic"
    image_path = root / f"{sample.name}_{suffix}.png"
    meta_path = root / f"{sample.name}_{suffix}.txt"
    generated.image.save(image_path)
    meta_lines = [
        f"anomaly_id: {generated.anomaly_id or ''}",
        f"method: {generated.method or ''}",
        f"prompt: {generated.prompt}",
        f"negative_prompt: {generated.negative_prompt}",
        f"seed: {generated.seed}",
    ]
    if generated.variation:
        meta_lines.append(f"variation_index: {generated.variation_index}")
        meta_lines.append(f"variation: {generated.variation}")
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")
    paths: dict[str, Path] = {"image": image_path, "metadata": meta_path}
    if generated.edit_mask is not None:
        mask_path = root / f"{sample.name}_{suffix}_edit_mask.png"
        Image.fromarray((generated.edit_mask.astype(np.uint8) * 255)).save(mask_path)
        paths["edit_mask"] = mask_path
    return paths


def save_annotation_artifact(
    name: str,
    annotation: AnnotationResult,
    output_dir: Path | str,
) -> Path:
    """Persist annotation overlay PNG."""
    root = Path(output_dir) / "annotations"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_annotated.png"
    Image.fromarray(annotation.overlay).save(path)
    return path


def show_judge_result(
    image: Image.Image | np.ndarray,
    judgment: Any,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure, Any]:
    """Show the judged RGB image with a scorecard (depth/seg are not used)."""
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    show_image(image, title="Judged image (RGB only)", ax=axes[0])
    axes[1].axis("off")
    lines = [
        f"decision: {getattr(judgment, 'decision', '?')}",
        f"overall: {getattr(judgment, 'overall', '?')}  (threshold gate)",
        f"prompt_faithfulness: {getattr(judgment, 'prompt_faithfulness', '?')}",
        f"physical_plausibility: {getattr(judgment, 'physical_plausibility', '?')}",
        f"annotation_correctness: {getattr(judgment, 'annotation_correctness', '?')}",
        f"edge_case_present: {getattr(judgment, 'edge_case_present', '?')}",
    ]
    gfid = getattr(judgment, "global_fidelity", None)
    ofid = getattr(judgment, "object_fidelity", None)
    if gfid is not None or ofid is not None:
        lines.append(f"global_fidelity: {gfid if gfid is not None else '?'}")
        lines.append(f"object_fidelity: {ofid if ofid is not None else '?'}")
    emb_g = getattr(judgment, "embed_real_sim_global", None)
    emb_n = getattr(judgment, "embed_neighbor_sim", None)
    if emb_g is not None or emb_n is not None:
        emb_l = getattr(judgment, "embed_real_sim_local", None)
        lines.append(
            f"embed KNN: real={emb_g if emb_g is not None else '?'}  "
            f"local={emb_l if emb_l is not None else '?'}  "
            f"neighbor={emb_n if emb_n is not None else '?'}",
        )
        reason = getattr(judgment, "embed_gate_reason", None)
        if reason:
            lines.append(f"embed_gate: {reason}")
    refs = getattr(judgment, "reference_paths", None) or []
    if refs:
        lines.append(f"references: {len(refs)}")
        for path in refs[:4]:
            lines.append(f"  - {Path(path).name}")
    elif getattr(judgment, "backend", "") == "api":
        lines.append("references: 0  (no same-class samples found)")
    lines.extend(
        [
            f"backend: {getattr(judgment, 'backend', '?')}",
            f"model: {getattr(judgment, 'model_id', '?')}",
            "",
            str(getattr(judgment, "rationale", "") or ""),
        ],
    )
    axes[1].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        wrap=True,
        transform=axes[1].transAxes,
    )
    fig.suptitle(
        title or f"VLM judge — {getattr(judgment, 'anomaly_id', '')}", fontsize=13
    )
    plt.tight_layout()
    return fig, axes


def save_judge_artifact(
    name: str,
    judgment: Any,
    output_dir: Path | str,
) -> Path:
    """Persist judge JSON next to other notebook artifacts."""
    pass

    root = Path(output_dir) / "judgments"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_judge.json"
    payload = judgment.to_dict() if hasattr(judgment, "to_dict") else dict(judgment)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


_OUTCOME_STYLE = {
    "accept": ("tab:green", "accepted"),
    "retry": ("tab:orange", "sent to retry"),
    "reject": ("tab:red", "rejected"),
    "surplus": ("tab:gray", "passed after target met (not exported)"),
}
_CLASS_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_fidelity_novelty(
    judged_rows: list[dict[str, Any]],
    *,
    min_real_sim: float,
    max_neighbor_sim: float,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 6.5),
) -> tuple[Figure, Axes]:
    """Scatter judged edits on the CLIP embedding fidelity/novelty proxies.

    x = cosine sim to the nearest *real* same-class image (fidelity proxy).
    y = cosine sim to the nearest image in real ∪ accepted synth (redundancy);
    the y axis is inverted so "up" means more novel. Dashed lines are the
    configured ``embedding_gate`` thresholds. Points inside the safe zone can
    still be retried/rejected by the VLM score, reference fidelity, box, or
    placement gates — this plot shows only the embedding signals.
    """
    rows = [
        r
        for r in judged_rows
        if r.get("embed_real_sim_global") is not None
        and r.get("embed_neighbor_sim") is not None
    ]
    fig, ax = plt.subplots(figsize=figsize)
    if not rows:
        ax.text(
            0.5,
            0.5,
            "No judged edits with embedding sims yet",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig, ax

    classes = sorted({str(r.get("anomaly_id", "?")) for r in rows})
    markers = {
        c: _CLASS_MARKERS[i % len(_CLASS_MARKERS)] for i, c in enumerate(classes)
    }
    for outcome, (color, label) in _OUTCOME_STYLE.items():
        for cls in classes:
            pts = [
                r
                for r in rows
                if r.get("outcome") == outcome and str(r.get("anomaly_id")) == cls
            ]
            if not pts:
                continue
            ax.scatter(
                [float(p["embed_real_sim_global"]) for p in pts],
                [float(p["embed_neighbor_sim"]) for p in pts],
                c=color,
                marker=markers[cls],
                s=28,
                alpha=0.7,
                edgecolors="none",
                label=f"{label} · {cls} (n={len(pts)})",
            )

    ax.axvline(min_real_sim, color="k", ls="--", lw=1)
    ax.axhline(max_neighbor_sim, color="k", ls="--", lw=1)
    ax.invert_yaxis()
    ax.text(
        min_real_sim,
        0.02,
        f"  min_real_sim_global={min_real_sim:.2f}\n  ← too far from real class images",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="bottom",
    )
    ax.text(
        0.99,
        max_neighbor_sim,
        f"max_neighbor_sim={max_neighbor_sim:.2f}  \nnear-duplicate below ↓  ",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        ha="right",
        va="top",
    )
    ax.text(
        0.98,
        0.97,
        "embedding safe zone",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
    )
    ax.set_xlabel("Fidelity proxy: CLIP cosine sim to nearest real same-class image  →")
    ax.set_ylabel(
        "Novelty proxy: CLIP cosine sim to nearest real ∪ accepted synth\n(axis inverted — up = more novel)"
    )
    ax.set_title(
        title
        or "Fidelity vs novelty of judged edits (practical embedding-based proxies)"
    )
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)
    fig.tight_layout()
    return fig, ax
