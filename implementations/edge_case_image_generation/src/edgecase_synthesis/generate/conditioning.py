"""Structure extraction: depth, segmentation, and config-driven edit masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
)


@dataclass
class DepthResult:
    depth_map: np.ndarray  # float32 (H, W) in [0, 1]
    colormap: np.ndarray  # uint8 RGB


@dataclass
class SegmentationResult:
    label_map: np.ndarray  # int (H, W)
    colored_map: np.ndarray  # uint8 RGB
    overlay: np.ndarray  # uint8 RGB
    num_regions: int
    edit_mask: np.ndarray | None = None  # optional bool ground mask from config classes


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_pil(image: Image.Image | np.ndarray | Path | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


def _palette_color(label: int) -> tuple[int, int, int]:
    """Deterministic RGB for a class id (no dataset-specific table)."""
    if label <= 0:
        return (0, 0, 0)
    rng = np.random.default_rng(label * 9973)
    color = rng.integers(40, 230, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label in np.unique(label_map):
        out[label_map == label] = _palette_color(int(label))
    return out


def _blend(rgb: np.ndarray, colored: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha, 0, 255).astype(
        np.uint8
    )


class DepthEstimator:
    """Monocular relative depth (model id from config)."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image | np.ndarray | Path | str) -> DepthResult:
        pil = _to_pil(image)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        depth = self.model(**inputs).predicted_depth.squeeze().float().cpu().numpy()
        h, w = pil.size[1], pil.size[0]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = float(depth.min()), float(depth.max())
        norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
        depth_u8 = (norm * 255).astype(np.uint8)
        colormap = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        return DepthResult(depth_map=norm.astype(np.float32), colormap=colormap)

    @classmethod
    def from_config(cls, cfg: Any, device: str | None = None):
        conditioning = cfg.get("conditioning", cfg)
        depth = conditioning.get("depth", conditioning)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(model_id=str(depth["model_id"]), device=device)


class Segmenter:
    """Semantic segmentation (model id + optional ground class ids from config)."""

    # Used when Mask2Former is configured but scipy is missing / invisible to transformers.
    _FALLBACK_SEGFORMER = "nvidia/segformer-b3-finetuned-ade-512-512"

    def __init__(
        self,
        model_name: str,
        *,
        ground_class_ids: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.ground_class_ids = list(ground_class_ids or [])
        self.device = resolve_device(device)
        self.processor = None
        self.model = None
        self._is_mask2former = "mask2former" in model_name.lower()

    @staticmethod
    def _scipy_ready() -> bool:
        """True only if scipy is importable *and* transformers agrees (it caches at import)."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            return False
        try:
            import transformers.utils.import_utils as iu

            # transformers sets `_scipy_available` once at import — refresh if needed.
            if not iu.is_scipy_available():
                available, version = iu._is_package_available("scipy")
                iu._scipy_available = available
                if hasattr(iu, "_scipy_version"):
                    iu._scipy_version = version
            return bool(iu.is_scipy_available())
        except Exception:
            return True

    def _ensure(self) -> None:
        if self.model is not None:
            return
        if self._is_mask2former and not self._scipy_ready():
            print(
                f"Mask2Former needs scipy (not visible in this kernel) — "
                f"falling back to {self._FALLBACK_SEGFORMER}",
                flush=True,
            )
            self.model_name = self._FALLBACK_SEGFORMER
            self._is_mask2former = False
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        if self._is_mask2former:
            from transformers import Mask2FormerForUniversalSegmentation

            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray | Path | str,
        *,
        overlay_alpha: float = 0.45,
        label_map_path: Path | str | None = None,
    ) -> SegmentationResult:
        pil = _to_pil(image)
        rgb = np.array(pil)
        h, w = rgb.shape[:2]

        if label_map_path is not None and Path(label_map_path).exists():
            labels = np.array(Image.open(label_map_path))
            if labels.ndim == 3:
                labels = labels[..., 0]
            if labels.shape != (h, w):
                labels = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            labels = labels.astype(np.int32)
        else:
            self._ensure()
            assert self.processor is not None and self.model is not None
            inputs = self.processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            if self._is_mask2former:
                labels_t = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(h, w)]
                )[0]
                labels = labels_t.cpu().numpy().astype(np.int32)
            else:
                logits = outputs.logits
                up = torch.nn.functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False
                )
                labels = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

        colored = _colorize_labels(labels)
        edit = None
        if self.ground_class_ids:
            edit = np.isin(labels, self.ground_class_ids)
        return SegmentationResult(
            label_map=labels,
            colored_map=colored,
            overlay=_blend(rgb, colored, alpha=overlay_alpha),
            num_regions=int(len(np.unique(labels))),
            edit_mask=edit,
        )

    @classmethod
    def from_config(cls, cfg: Any, device: str | None = None):
        conditioning = cfg.get("conditioning", cfg)
        seg = conditioning.get("segmentation", conditioning)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            model_name=str(seg["model_name"]),
            ground_class_ids=list(seg.get("ground_class_ids") or []),
            device=device,
        )


def build_anomaly_edit_mask(
    segmentation: SegmentationResult | None,
    edit_mask_cfg: dict | None,
    *,
    width: int,
    height: int,
    depth: DepthResult | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edit mask from YAML — no dataset-specific geometry in code.

    Modes (``edit_mask.mode``):
      - ``ellipse`` — cx, cy, rx, ry as fractions of width/height
      - ``rect`` — x0, y0, x1, y1 as fractions
      - ``full`` — whole image
      - ``road_patch`` — place an ellipse on seg road support (lower band + near depth)
      - ``seg_intersection`` — ellipse/rect intersected with seg.edit_mask or class ids
    """
    cfg = dict(edit_mask_cfg or {})
    mode = str(cfg.get("mode", "ellipse")).lower()
    blur_sigma = float(cfg.get("blur_sigma", 1.5))
    yy, xx = np.mgrid[0:height, 0:width]

    if mode == "full":
        mask = np.ones((height, width), dtype=bool)
    elif mode == "rect":
        x0 = float(cfg.get("x0", 0.25)) * width
        x1 = float(cfg.get("x1", 0.75)) * width
        y0 = float(cfg.get("y0", 0.25)) * height
        y1 = float(cfg.get("y1", 0.75)) * height
        mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    elif mode == "road_patch":
        mask = _road_patch_ellipse(
            segmentation, depth, width=width, height=height, cfg=cfg, xx=xx, yy=yy
        )
    else:  # ellipse (default) and seg_intersection base
        cx = float(cfg.get("cx", 0.5)) * width
        cy = float(cfg.get("cy", 0.5)) * height
        rx = float(cfg.get("rx", 0.2)) * width
        ry = float(cfg.get("ry", 0.2)) * height
        mask = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0

    if mode == "seg_intersection" or bool(cfg.get("intersect_seg", False)):
        support = _seg_support(segmentation, width, height, cfg)
        clipped = mask & support
        # Keep the geometric prior if seg support wipes almost everything.
        if float(clipped.mean()) >= max(0.001, 0.15 * float(mask.mean() + 1e-8)):
            mask = clipped

    seed_mask = mask.copy()

    # Optional: prefer nearer pixels when depth is available (generic, not domain-specific).
    if depth is not None and bool(cfg.get("prefer_near", False)) and mask.any():
        d = depth.depth_map
        if d.shape != (height, width):
            d = cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR)
        # Keep the nearer ~70% of the seed mask (quantile 0.30), not a thin near band.
        q = float(cfg.get("prefer_near_quantile", 0.30))
        near = d >= float(np.quantile(d[mask], q))
        clipped = mask & near
        # If prefer_near collapses to a crescent/sliver, keep the seed ellipse.
        min_keep = float(cfg.get("min_mask_keep", 0.35))
        if float(clipped.mean()) >= min_keep * float(seed_mask.mean() + 1e-8):
            mask = clipped
        else:
            mask = seed_mask

    mask = _dilate(mask, int(cfg.get("dilate", 1)))
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(blur_sigma, 0.5))
    return mask, np.clip(weight, 0.0, 1.0)


def _road_patch_ellipse(
    segmentation: SegmentationResult | None,
    depth: DepthResult | None,
    *,
    width: int,
    height: int,
    cfg: dict,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Anchor an ellipse on road pixels in the lower frame (dashcam-friendly)."""
    support = _seg_support(segmentation, width, height, cfg)
    y_min = float(cfg.get("y_min", 0.48))
    y_max = float(cfg.get("y_max", 0.92))
    x_min = float(cfg.get("x_min", 0.22))
    x_max = float(cfg.get("x_max", 0.78))
    band = (
        (yy >= y_min * height)
        & (yy <= y_max * height)
        & (xx >= x_min * width)
        & (xx <= x_max * width)
    )
    region = support & band
    if not region.any():
        region = support & (yy >= y_min * height) & (yy <= y_max * height)
    if not region.any():
        # No road seg — fall back to fixed prior ellipse.
        cx = float(cfg.get("cx", 0.5)) * width
        cy = float(cfg.get("cy", 0.75)) * height
    else:
        ys, xs = np.where(region)
        prior_cx = float(cfg.get("cx", 0.5)) * width
        prior_cy = float(cfg.get("cy", 0.75)) * height
        dist = np.sqrt((xs.astype(np.float32) - prior_cx) ** 2 + (ys.astype(np.float32) - prior_cy) ** 2)
        dist_n = dist / (float(dist.max()) + 1e-6)
        if depth is not None:
            d = depth.depth_map
            if d.shape != (height, width):
                d = cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR)
            near = d[ys, xs].astype(np.float32)
            near_n = (near - near.min()) / (float(near.max() - near.min()) + 1e-6)
            score = 0.65 * near_n + 0.35 * (1.0 - dist_n)
        else:
            score = 1.0 - dist_n
        best = int(np.argmax(score))
        cx = float(xs[best])
        cy = float(ys[best])

    rx = float(cfg.get("rx", 0.10)) * width
    ry = float(cfg.get("ry", 0.12)) * height
    return ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0


def _seg_support(
    segmentation: SegmentationResult | None,
    width: int,
    height: int,
    cfg: dict,
) -> np.ndarray:
    if segmentation is None:
        return np.ones((height, width), dtype=bool)
    class_ids = cfg.get("seg_class_ids")
    if class_ids and segmentation.label_map is not None:
        labels = segmentation.label_map
        if labels.shape != (height, width):
            labels = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        return np.isin(labels, list(class_ids))
    if segmentation.edit_mask is not None:
        mask = segmentation.edit_mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
    return np.ones((height, width), dtype=bool)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.dilate(mask.astype(np.uint8), k, iterations=iterations)
    return out.astype(bool)
