"""Visualization helpers for notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from edgecase_synthesis.annotation import AnnotationResult
from edgecase_synthesis.conditioning import DepthResult, SegmentationResult
from edgecase_synthesis.data import ImageSample
from edgecase_synthesis.generation import GenerationResult


def show_image(image: Image.Image | np.ndarray, *, title: str = "", ax=None):
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


def show_samples(samples: list[ImageSample], *, ncol: int = 3, figsize=(12, 4)):
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
    figsize=(12, 4),
):
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
    figsize=(12, 4),
):
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
    figsize=(14, 8),
):
    """2×2 panel: original, depth, segmentation, overlay — for notebook summaries."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    show_image(sample.image, title="1. Real image", ax=axes[0, 0])
    show_image(depth.colormap, title="2. Depth (ControlNet conditioning)", ax=axes[0, 1])
    show_image(seg.colored_map, title="3. Segmentation map", ax=axes[1, 0])
    show_image(seg.overlay, title="4. Segmentation overlay", ax=axes[1, 1])

    fig.suptitle(
        f"Structure extraction — {sample.name}\n"
        "(these maps guide diffusion; they are NOT sent to the VLM judge later)",
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
    figsize=(12, 5),
):
    """Original vs ControlNet edit with the prompt used."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    show_image(sample.image, title="Source", ax=axes[0])
    title = "Synthetic edit"
    if generated.anomaly_id:
        method = generated.method or ""
        title = f"{generated.anomaly_id}" + (f" [{method}]" if method else "")
    show_image(generated.image, title=title, ax=axes[1])
    prompt_preview = generated.prompt if len(generated.prompt) < 140 else generated.prompt[:137] + "..."
    fig.suptitle(
        f"Anomaly edit — {sample.name}\nPrompt: {prompt_preview}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig, axes


def show_annotation_result(
    image: Image.Image | np.ndarray,
    annotation: AnnotationResult,
    *,
    title: str = "Annotations",
    figsize=(12, 5),
):
    """Original vs open-vocabulary detections + SAM masks."""
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
    meta_path.write_text(
        "\n".join(
            [
                f"anomaly_id: {generated.anomaly_id or ''}",
                f"method: {generated.method or ''}",
                f"prompt: {generated.prompt}",
                f"negative_prompt: {generated.negative_prompt}",
                f"seed: {generated.seed}",
            ]
        ),
        encoding="utf-8",
    )
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
    figsize=(10, 4),
):
    """Show the judged RGB image with a scorecard (depth/seg are not used)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]})
    show_image(image, title="Judged image (RGB only)", ax=axes[0])
    axes[1].axis("off")
    lines = [
        f"decision: {getattr(judgment, 'decision', '?')}",
        f"overall: {getattr(judgment, 'overall', '?')}  (threshold gate)",
        f"prompt_faithfulness: {getattr(judgment, 'prompt_faithfulness', '?')}",
        f"physical_plausibility: {getattr(judgment, 'physical_plausibility', '?')}",
        f"annotation_correctness: {getattr(judgment, 'annotation_correctness', '?')}",
        f"edge_case_present: {getattr(judgment, 'edge_case_present', '?')}",
        f"backend: {getattr(judgment, 'backend', '?')}",
        f"model: {getattr(judgment, 'model_id', '?')}",
        "",
        str(getattr(judgment, "rationale", "") or ""),
    ]
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
    fig.suptitle(title or f"VLM judge — {getattr(judgment, 'anomaly_id', '')}", fontsize=13)
    plt.tight_layout()
    return fig, axes


def save_judge_artifact(
    name: str,
    judgment: Any,
    output_dir: Path | str,
) -> Path:
    """Persist judge JSON next to other notebook artifacts."""
    import json

    root = Path(output_dir) / "judgments"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_judge.json"
    payload = judgment.to_dict() if hasattr(judgment, "to_dict") else dict(judgment)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
