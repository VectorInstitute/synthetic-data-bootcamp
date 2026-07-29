"""Thin single-image synthesis loop for Notebook 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf
from PIL import Image

from edgecase_synthesis.compare_methods import COMPARE_METHODS, METHOD_SPECS, MethodComparer
from edgecase_synthesis.conditioning import DepthResult, SegmentationResult
from edgecase_synthesis.config import load_anomaly, merge_generation_anomaly
from edgecase_synthesis.generation import GenerationResult


@dataclass
class SynthesisResult:
    """One anomaly edit with the method that produced it."""

    anomaly_id: str
    method: str
    generated: GenerationResult


def validate_method(method: str) -> str:
    method = str(method).lower()
    if method not in COMPARE_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {COMPARE_METHODS}")
    return method


def default_method_map(cfg: Any) -> dict[str, str]:
    """Read method_by_anomaly from the active dataset package."""
    dataset = cfg.get("dataset") if hasattr(cfg, "get") else None
    raw = {}
    if dataset is not None:
        raw = OmegaConf.to_container(dataset.get("method_by_anomaly") or {}, resolve=True) or {}
    return {str(k): str(v) for k, v in dict(raw).items()}


def resolve_method_map(
    method_by_anomaly: dict[str, str] | None,
    workshop_anomalies: list[str],
    *,
    cfg: Any = None,
) -> dict[str, str]:
    """Fill missing anomalies from dataset.method_by_anomaly (or inpaint)."""
    defaults = default_method_map(cfg) if cfg is not None else {}
    fallback = "inpaint"
    if cfg is not None:
        fallback = str(cfg.generation.get("default_anomaly_method") or fallback)
    out: dict[str, str] = {}
    provided = method_by_anomaly or {}
    for anomaly_id in workshop_anomalies:
        out[anomaly_id] = validate_method(
            provided.get(anomaly_id, defaults.get(anomaly_id, fallback))
        )
    return out


def synthesize_one(
    image: Image.Image,
    *,
    anomaly_id: str,
    method: str,
    cfg: Any,
    depth: DepthResult,
    segmentation: SegmentationResult,
    comparer: MethodComparer,
    project_root: Any = None,
    seed_offset: int = 0,
) -> SynthesisResult:
    method = validate_method(method)
    dataset = str(cfg.dataset_name)
    anomaly_cfg = load_anomaly(dataset, anomaly_id, start=project_root)
    generated = comparer.run_method(
        method,
        image,
        depth=depth,
        segmentation=segmentation,
        generation_cfg=cfg.generation,
        anomaly_cfg=anomaly_cfg,
        seed_offset=seed_offset,
    )
    return SynthesisResult(anomaly_id=anomaly_id, method=method, generated=generated)


def merged_prompt(cfg: Any, anomaly_id: str, *, method: str | None = None, project_root: Any = None) -> str:
    dataset = str(cfg.dataset_name)
    anomaly_cfg = load_anomaly(dataset, anomaly_id, start=project_root)
    merged = merge_generation_anomaly(cfg.generation, anomaly_cfg, method=method)
    return str(merged.get("prompt", ""))
