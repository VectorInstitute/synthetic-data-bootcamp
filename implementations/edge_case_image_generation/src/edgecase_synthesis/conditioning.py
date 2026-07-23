"""Structure extraction: depth maps and ADE20K-compatible segmentation."""

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

# ADE20K / ControlNet-seg color protocol (150 classes). Random SAM colors break
# sd-controlnet-seg; this palette is what that ControlNet was trained on.
ADE20K_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ],
    dtype=np.uint8,
)

# ADE20K SceneParse150 indices (SegFormer) used as editable ground for snow.
ADE20K_TRACK_CLASSES = {
    6,  # road
    9,  # grass
    10,  # sidewalk
    12,  # earth
}


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DepthResult:
    """Depth estimation output for one image."""

    depth_map: np.ndarray  # float32, shape (H, W), values in [0, 1]
    colormap: np.ndarray  # uint8 RGB visualization, shape (H, W, 3)


@dataclass
class SegmentationResult:
    """Segmentation output for one image."""

    masks: np.ndarray  # bool, shape (N, H, W) — one mask per kept region/class
    colored_map: np.ndarray  # uint8 RGB ADE20K palette, shape (H, W, 3)
    overlay: np.ndarray  # uint8 RGB blend with original, shape (H, W, 3)
    num_regions: int
    label_map: np.ndarray | None = None  # int class ids (H, W), if available
    edit_mask: np.ndarray | None = None  # bool (H, W) track/ground for localized edits
    # Soft weights for more realistic snow (0–1). Prefer this over a flat bool mask.
    snow_weight: np.ndarray | None = None  # float32 (H, W)
    winter_weight: np.ndarray | None = None  # float32 (H, W) terrain/veg frost grade


class DepthEstimator:
    """Depth Anything V2 via Hugging Face Transformers — no API key needed."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image | np.ndarray | Path | str) -> DepthResult:
        pil_image = _to_pil(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth.squeeze().float().cpu().numpy()

        # Resize depth to the original image size for ControlNet alignment.
        h, w = pil_image.size[1], pil_image.size[0]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        depth_min, depth_max = float(depth.min()), float(depth.max())
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)

        depth_u8 = (depth_norm * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        return DepthResult(depth_map=depth_norm.astype(np.float32), colormap=colormap)

    @classmethod
    def from_config(cls, cfg: dict | Any, device: str | None = None):
        conditioning = cfg.get("conditioning", cfg)
        depth = conditioning.get("depth", conditioning)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            model_id=depth.get("model_id", "depth-anything/Depth-Anything-V2-Base-hf"),
            device=device,
        )


class Segmenter:
    """ADE20K semantic segmentation for ControlNet-seg + track edit masks.

    Uses a small SegFormer by default (laptop-friendly). Unlike MobileSAM's
    random instance colors, outputs use the ADE20K palette that
    ``sd-controlnet-seg`` expects.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.processor = None
        self.model = None

    @classmethod
    def from_config(cls, cfg: dict | Any, device: str | None = None):
        conditioning = cfg.get("conditioning", cfg)
        seg = conditioning.get("segmentation", conditioning)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            model_name=seg.get(
                "model_name", "nvidia/segformer-b0-finetuned-ade-512-512"
            ),
            device=device,
        )

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
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
        pil_image = _to_pil(image)
        rgb = np.array(pil_image)
        h, w = rgb.shape[:2]

        if label_map_path is not None and Path(label_map_path).exists():
            return self._from_railsem19_labels(
                rgb,
                Path(label_map_path),
                overlay_alpha=overlay_alpha,
            )

        self._ensure_model()
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        label_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        colored_map = _colorize_ade20k(label_map)
        edit_mask = _edit_mask_from_ade20k(label_map)
        masks = _masks_from_label_map(label_map)
        overlay = _blend(rgb, colored_map, alpha=overlay_alpha)

        return SegmentationResult(
            masks=masks,
            colored_map=colored_map,
            overlay=overlay,
            num_regions=int(len(np.unique(label_map))),
            label_map=label_map,
            edit_mask=edit_mask,
            snow_weight=edit_mask.astype(np.float32) * 0.55,
            winter_weight=_winter_weight_from_ade20k(label_map),
        )

    def _from_railsem19_labels(
        self,
        rgb: np.ndarray,
        label_path: Path,
        *,
        overlay_alpha: float,
    ) -> SegmentationResult:
        """Use RailSem19 uint8 GT labels when available (best track masks)."""
        labels = np.array(Image.open(label_path))
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        if labels.shape[:2] != rgb.shape[:2]:
            labels = cv2.resize(
                labels,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Map RS19 ids → ADE20K-ish colors for ControlNet-seg compatibility.
        # RS19: rail-track=12, trackbed=15, rail-raised=17, tram-track=3, terrain=9, sky=10, veg=8
        rs19_to_ade = {
            0: 6,  # road → road
            1: 10,  # sidewalk → sidewalk
            3: 6,  # tram-track → road
            5: 93,  # pole → pole (ADE20K)
            8: 4,  # vegetation → tree
            9: 12,  # terrain → earth
            10: 2,  # sky → sky
            12: 6,  # rail-track → road (editable ground plane)
            13: 20,  # car
            15: 12,  # trackbed → earth
            17: 6,  # rail-raised → road
            255: 0,  # void
        }
        ade_map = np.zeros_like(labels, dtype=np.int32)
        for src, dst in rs19_to_ade.items():
            ade_map[labels == src] = dst

        colored_map = _colorize_ade20k(ade_map)
        # Concentrate snow on thin ribbons along the rails; only a light dusting
        # on ballast so the dual-track corridor doesn't become a white slab.
        rail_core = ((labels == 17) | (labels == 3)).astype(np.uint8)
        if float(rail_core.mean()) < 0.002:
            rail_core = (labels == 12).astype(np.uint8)
        rail_ribbon = cv2.dilate(rail_core, np.ones((5, 5), np.uint8), iterations=2).astype(np.float32)
        rail_ribbon = cv2.GaussianBlur(rail_ribbon, (0, 0), 1.5)

        snow_weight = np.zeros(labels.shape, dtype=np.float32)
        snow_weight = np.maximum(snow_weight, rail_ribbon * 0.50)
        # Very light ballast dusting only — gravel must stay readable.
        snow_weight[labels == 15] = np.maximum(snow_weight[labels == 15], 0.05)
        snow_weight[labels == 12] = np.maximum(snow_weight[labels == 12], 0.04)
        snow_weight = cv2.GaussianBlur(snow_weight, (0, 0), 1.0)

        winter_weight = np.zeros(labels.shape, dtype=np.float32)
        winter_weight[labels == 9] = 0.90  # terrain
        winter_weight[labels == 8] = 0.85  # vegetation
        winter_weight[labels == 0] = 0.35  # road
        winter_weight = cv2.GaussianBlur(winter_weight, (0, 0), 3)

        edit_mask = snow_weight > 0.05
        if edit_mask.mean() < 0.01:
            edit_mask = _edit_mask_from_ade20k(ade_map)
            snow_weight = edit_mask.astype(np.float32) * 0.45

        masks = _masks_from_label_map(labels.astype(np.int32))
        overlay = _blend(rgb, colored_map, alpha=overlay_alpha)
        return SegmentationResult(
            masks=masks,
            colored_map=colored_map,
            overlay=overlay,
            num_regions=int(len(np.unique(labels))),
            label_map=labels.astype(np.int32),
            edit_mask=edit_mask,
            snow_weight=snow_weight,
            winter_weight=winter_weight,
        )


def build_anomaly_edit_mask(
    segmentation: SegmentationResult | None,
    edit_mask_cfg: dict | None,
    *,
    width: int,
    height: int,
    depth: DepthResult | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a boolean edit mask + soft weight for localized anomaly inserts.

    Modes (from anomaly YAML ``edit_mask.mode``):

    - ``track_blob`` / ``auto`` — place a compact ellipse on the *detected* near
      track corridor (depth + cab-view geometry; seg optional). No per-image
      ``x_center`` required — scales across frames.
    - ``track_strip`` — horizontal band across the detected track at an
      auto-chosen near depth band (fallen branch / debris).
    - ``blob`` / ``strip`` — legacy fixed normalized fractions (avoid for batch).
    - ``snow`` — soft rail/trackbed weights from segmentation (weather path).

    Returns
    -------
    edit_mask : bool (H, W)
    edit_weight : float32 (H, W) in [0, 1] for priming / compositing
    """
    cfg = dict(edit_mask_cfg or {})
    mode = str(cfg.get("mode", "track_blob")).lower()
    blur_sigma = float(cfg.get("blur_sigma", 2.0))

    if mode == "snow":
        weight = _snow_weight_at_size(segmentation, width, height)
        mask = weight > 0.05
        return mask, weight

    if mode in {"track_blob", "auto", "dynamic_blob"}:
        mask = _dynamic_track_blob(segmentation, depth, width, height, cfg)
    elif mode in {"track_strip", "dynamic_strip"}:
        mask = _dynamic_track_strip(segmentation, depth, width, height, cfg)
    elif mode == "strip":
        yy, xx = np.mgrid[0:height, 0:width]
        y0 = float(cfg.get("y_start", 0.60))
        y1 = float(cfg.get("y_end", 0.78))
        x_margin = float(cfg.get("x_margin", 0.28))
        mask = (yy >= height * y0) & (yy <= height * y1)
        mask &= (xx >= width * x_margin) & (xx <= width * (1.0 - x_margin))
        if bool(cfg.get("intersect_track", False)):
            track = _track_support_mask(segmentation, width, height, depth=depth)
            clipped = mask & track
            if float(clipped.mean()) >= 0.002:
                mask = clipped
    else:  # legacy blob with fixed fractions
        yy, xx = np.mgrid[0:height, 0:width]
        y0 = float(cfg.get("y_start", 0.58))
        y1 = float(cfg.get("y_end", 0.88))
        cx = float(cfg.get("x_center", 0.48)) * width
        rx = float(cfg.get("x_radius", 0.07)) * width
        ry = float(cfg.get("y_radius", 0.10)) * height
        cy = 0.5 * (y0 + y1) * height
        mask = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
        if bool(cfg.get("intersect_track", False)):
            track = _track_support_mask(segmentation, width, height, depth=depth)
            clipped = mask & track
            if float(clipped.mean()) >= 0.002:
                mask = clipped

    mask = _dilate_mask(mask.astype(bool), iterations=1)
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(blur_sigma, 0.5))
    weight = np.clip(weight, 0.0, 1.0)
    return mask, weight


def _cab_track_corridor(height: int, width: int) -> np.ndarray:
    """Perspective trapezoid where forward-facing cab rails usually sit."""
    yy, xx = np.mgrid[0:height, 0:width]
    # Narrow near the horizon, wider toward the camera.
    t = np.clip((yy / max(height - 1, 1) - 0.30) / 0.70, 0.0, 1.0)
    half = width * (0.06 + 0.40 * (t**1.15))
    return (yy > height * 0.38) & (np.abs(xx - width * 0.5) < half)


def _depth_nearness(depth: DepthResult | None, width: int, height: int) -> np.ndarray | None:
    """Return float map in [0, 1] where larger ≈ closer to camera, or None."""
    if depth is None or depth.depth_map is None:
        return None
    d = depth.depth_map.astype(np.float32)
    if d.shape[:2] != (height, width):
        d = cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR)
    # Depth Anything V2 relative depth: after min-max, warmer/brighter ≈ nearer
    # in our viz — treat higher values as nearer.
    lo, hi = float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))
    if hi <= lo:
        return np.zeros((height, width), dtype=np.float32)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def _track_score_map(
    segmentation: SegmentationResult | None,
    depth: DepthResult | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Per-pixel score for 'good place to drop a track anomaly'."""
    corridor = _cab_track_corridor(height, width).astype(np.float32)
    near = _depth_nearness(depth, width, height)
    score = corridor.copy()
    if near is not None:
        # Prefer near ground inside the corridor (not far vanishing point).
        score *= 0.35 + 0.65 * near
    else:
        yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        score *= 0.40 + 0.60 * np.broadcast_to(yy, (height, width))

    track = _track_support_mask(segmentation, width, height, depth=depth)
    if float(track.mean()) > 0.01:
        # Boost semantic / GT track when available; never rely on it alone
        # (ADE20K has no rail class — often fragmented on Nordland).
        score = score * (0.55 + 0.45 * track.astype(np.float32))

    return score


def _pick_track_anchor(
    score: np.ndarray,
    *,
    y_lo: float = 0.52,
    y_hi: float = 0.88,
) -> tuple[float, float, float]:
    """Return (cx, cy, local_half_width) in pixel coords from a score map."""
    h, w = score.shape
    yy, xx = np.mgrid[0:h, 0:w]
    band = (yy >= h * y_lo) & (yy <= h * y_hi) & (score > 0)
    region = score * band.astype(np.float32)
    if float(region.max()) <= 1e-6:
        region = score
        band = score > 0

    # Soft threshold → mass for a stable centroid across frames.
    thr = float(np.quantile(region[band], 0.70)) if band.any() else 0.0
    mass = region >= max(thr, 1e-6)
    if not mass.any():
        return 0.5 * w, 0.72 * h, 0.10 * w

    weights = region[mass]
    cy = float(np.average(yy[mass], weights=weights))
    cx = float(np.average(xx[mass], weights=weights))

    # Local track half-width from corridor width at this row.
    row = int(np.clip(round(cy), 0, h - 1))
    row_mask = mass[row]
    if row_mask.any():
        xs = np.where(row_mask)[0]
        half = 0.5 * float(xs.max() - xs.min() + 1)
    else:
        half = 0.12 * w
    half = float(np.clip(half, 0.06 * w, 0.28 * w))
    return cx, cy, half


def _dynamic_track_blob(
    segmentation: SegmentationResult | None,
    depth: DepthResult | None,
    width: int,
    height: int,
    cfg: dict,
) -> np.ndarray:
    """Ellipse on the auto-detected near track — no manual x/y per image."""
    score = _track_score_map(segmentation, depth, width, height)
    y_lo = float(cfg.get("y_lo", 0.52))
    y_hi = float(cfg.get("y_hi", 0.90))
    cx, cy, half = _pick_track_anchor(score, y_lo=y_lo, y_hi=y_hi)
    size_frac = float(cfg.get("size_frac", 0.55))
    aspect = float(cfg.get("aspect", 1.35))  # taller than wide for upright objects
    rx = max(4.0, half * size_frac)
    ry = max(4.0, rx * aspect)
    yy, xx = np.mgrid[0:height, 0:width]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _dynamic_track_strip(
    segmentation: SegmentationResult | None,
    depth: DepthResult | None,
    width: int,
    height: int,
    cfg: dict,
) -> np.ndarray:
    """Horizontal debris strip across the auto-detected track corridor."""
    score = _track_score_map(segmentation, depth, width, height)
    y_lo = float(cfg.get("y_lo", 0.55))
    y_hi = float(cfg.get("y_hi", 0.88))
    cx, cy, half = _pick_track_anchor(score, y_lo=y_lo, y_hi=y_hi)
    thickness = float(cfg.get("thickness_frac", 0.35)) * half
    thickness = max(3.0, thickness)
    # Span most of the local track width.
    span = float(cfg.get("span_frac", 1.6)) * half
    yy, xx = np.mgrid[0:height, 0:width]
    return (np.abs(yy - cy) <= thickness) & (np.abs(xx - cx) <= span)


def _track_support_mask(
    segmentation: SegmentationResult | None,
    width: int,
    height: int,
    depth: DepthResult | None = None,
) -> np.ndarray:
    """Pixels that look like track / trackbed (or a cab-view prior)."""
    if segmentation is not None and segmentation.label_map is not None:
        labels = segmentation.label_map
        # RailSem19 ids when GT was used; ADE ids otherwise.
        track = np.isin(labels, [3, 6, 12, 15, 17]).astype(np.uint8)
        if track.shape != (height, width):
            track = cv2.resize(track, (width, height), interpolation=cv2.INTER_NEAREST)
        if float(track.mean()) > 0.01:
            return track.astype(bool)

    if segmentation is not None and segmentation.edit_mask is not None:
        mask = segmentation.edit_mask.astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if float(mask.mean()) > 0.01:
            return mask.astype(bool)

    corridor = _cab_track_corridor(height, width)
    near = _depth_nearness(depth, width, height)
    if near is not None:
        return corridor & (near > 0.35)
    return corridor


def _snow_weight_at_size(
    segmentation: SegmentationResult | None,
    width: int,
    height: int,
) -> np.ndarray:
    if segmentation is not None and segmentation.snow_weight is not None:
        weight = segmentation.snow_weight.astype(np.float32)
    elif segmentation is not None and segmentation.edit_mask is not None:
        weight = segmentation.edit_mask.astype(np.float32) * 0.55
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        prior = (
            (yy > height * 0.35)
            & (np.abs(xx - width / 2) < (width * 0.12 + (yy / height) * width * 0.38))
        )
        weight = prior.astype(np.float32) * 0.5

    if weight.shape[:2] != (height, width):
        weight = cv2.resize(weight, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.GaussianBlur(weight, (0, 0), 1.5), 0.0, 1.0)


def find_railsem19_label_map(image_path: Path | str, dataset_root: Path | str | None = None) -> Path | None:
    """Locate the uint8 label PNG for a RailSem19 sample image."""
    path = Path(image_path)
    stem = path.stem
    # samples are named rs19_rs00000.jpg → frame id rs00000
    frame_id = stem.replace("rs19_", "") if stem.startswith("rs19_") else stem

    candidates: list[Path] = []
    if dataset_root is not None:
        root = Path(dataset_root)
        candidates.append(root / "uint8" / "rs19_val" / f"{frame_id}.png")
    # Walk up from samples toward data/rs19_val
    for parent in [path.parent, *path.parents]:
        candidates.append(parent / "rs19_val" / "uint8" / "rs19_val" / f"{frame_id}.png")
        candidates.append(parent / "uint8" / "rs19_val" / f"{frame_id}.png")
        if parent.name == "EdgeCaseSynthesis" or (parent / "data").exists():
            candidates.append(parent / "data" / "rs19_val" / "uint8" / "rs19_val" / f"{frame_id}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _colorize_ade20k(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = min(int(label_map.max()), len(ADE20K_PALETTE) - 1)
    for label in range(max_id + 1):
        colored[label_map == label] = ADE20K_PALETTE[label]
    return colored


def _edit_mask_from_ade20k(label_map: np.ndarray) -> np.ndarray:
    """Boolean mask of ground-like classes, with a cab-view center prior."""
    h, w = label_map.shape
    semantic = np.isin(label_map, list(ADE20K_TRACK_CLASSES))

    # Cab-view prior: lower-center trapezoid where rails usually sit.
    yy, xx = np.mgrid[0:h, 0:w]
    prior = (yy > h * 0.35) & (np.abs(xx - w / 2) < (w * 0.15 + (yy / h) * w * 0.35))
    mask = semantic | prior
    # Prefer intersection when semantic fired enough; else fall back to prior.
    if semantic.mean() > 0.02:
        mask = semantic & prior
        if mask.mean() < 0.01:
            mask = semantic | prior
    return _dilate_mask(mask.astype(bool), iterations=2)


def _winter_weight_from_ade20k(label_map: np.ndarray) -> np.ndarray:
    """Soft weights for global winter grade (terrain / vegetation / road)."""
    weight = np.zeros(label_map.shape, dtype=np.float32)
    weight[np.isin(label_map, [9, 12])] = 0.5  # grass / earth
    weight[np.isin(label_map, [4, 17])] = 0.35  # tree / plant
    weight[label_map == 6] = 0.25  # road
    return cv2.GaussianBlur(weight, (0, 0), 3)


def _masks_from_label_map(label_map: np.ndarray) -> np.ndarray:
    ids = [i for i in np.unique(label_map) if i != 255]
    if not ids:
        return np.zeros((0, *label_map.shape), dtype=bool)
    return np.stack([(label_map == i) for i in ids], axis=0).astype(bool)


def _dilate_mask(mask: np.ndarray, *, iterations: int = 3) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated.astype(bool)


def _to_pil(image: Image.Image | np.ndarray | Path | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        from edgecase_synthesis.data import load_image

        return load_image(Path(image))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _blend(base: np.ndarray, overlay: np.ndarray, *, alpha: float) -> np.ndarray:
    return np.clip(
        base.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
