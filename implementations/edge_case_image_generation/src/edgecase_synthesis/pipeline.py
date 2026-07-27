"""Thin single-image synthesis loop for Notebook 1.

Notebook 1.5 owns method *comparison*. Here the learner picks one method per
anomaly (e.g. ``traffic_cone → inpaint``) and runs load → edit → annotate → judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf
from PIL import Image

from edgecase_synthesis.compare_methods import COMPARE_METHODS, METHOD_SPECS, MethodComparer
from edgecase_synthesis.conditioning import DepthResult, SegmentationResult
from edgecase_synthesis.config import load_anomaly, merge_generation_anomaly
from edgecase_synthesis.generation import GenerationResult


# Sensible defaults after Notebook 1.5 — override in the notebook.
DEFAULT_METHOD_BY_ANOMALY: dict[str, str] = {
    "pothole": "inpaint",
    "traffic_cone": "inpaint",
    "fog": "instruct",
    "ground_animal": "inpaint",
}


@dataclass
class SynthesisResult:
    """One anomaly edit with the method that produced it."""

    anomaly_id: str
    method: str
    generated: GenerationResult


def validate_method(method: str) -> str:
    key = str(method).lower().strip()
    if key not in COMPARE_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {COMPARE_METHODS}")
    return key


def resolve_method_map(
    method_by_anomaly: dict[str, str] | None,
    workshop_anomalies: list[str],
) -> dict[str, str]:
    """Fill missing anomalies from DEFAULT_METHOD_BY_ANOMALY."""
    out: dict[str, str] = {}
    for anomaly_id in workshop_anomalies:
        raw = (method_by_anomaly or {}).get(
            anomaly_id, DEFAULT_METHOD_BY_ANOMALY.get(anomaly_id, "inpaint")
        )
        out[anomaly_id] = validate_method(raw)
    return out


def synthesize_one(
    image: Image.Image,
    *,
    anomaly_id: str,
    method: str,
    cfg: Any,
    comparer: MethodComparer,
    depth: DepthResult,
    segmentation: SegmentationResult,
    project_root: Any = None,
    seed_offset: int = 0,
) -> SynthesisResult:
    """Run one anomaly edit with the chosen method (from Notebook 1.5)."""
    method = validate_method(method)
    dataset = str(cfg.generation.anomaly_dataset)
    anomaly_cfg = load_anomaly(dataset, anomaly_id, start=project_root)

    if seed_offset:
        anomaly_cfg = OmegaConf.create(OmegaConf.to_container(anomaly_cfg, resolve=True))
        OmegaConf.set_struct(anomaly_cfg, False)
        base_seed = int(cfg.generation.get("seed", 42))
        anomaly_cfg.seed = base_seed + int(seed_offset)

    generated = comparer.run_method(
        method,
        image,
        depth=depth,
        segmentation=segmentation,
        generation_cfg=cfg.generation,
        anomaly_cfg=anomaly_cfg,
    )
    return SynthesisResult(anomaly_id=anomaly_id, method=method, generated=generated)


def method_blurb(method: str) -> str:
    return METHOD_SPECS[validate_method(method)].summary


def merged_prompt(cfg: Any, anomaly_id: str, *, project_root: Any = None) -> str:
    dataset = str(cfg.generation.anomaly_dataset)
    anomaly_cfg = load_anomaly(dataset, anomaly_id, start=project_root)
    merged = merge_generation_anomaly(cfg.generation, anomaly_cfg)
    return str(merged.get("prompt", ""))
