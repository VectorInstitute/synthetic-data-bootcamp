"""Heuristics for rejecting unusable source images (tunnels, black frames, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Summary statistics used by the contrast gate."""

    luminance_mean: float
    luminance_std: float
    luminance_range: float


def image_quality_metrics(image: Image.Image | np.ndarray) -> ImageQualityMetrics:
    """Compute luminance statistics on an RGB image."""
    rgb = _to_rgb_array(image)
    luminance = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    )
    return ImageQualityMetrics(
        luminance_mean=float(luminance.mean()),
        luminance_std=float(luminance.std()),
        luminance_range=float(luminance.max() - luminance.min()),
    )


def is_usable_sample_image(
    image: Image.Image | np.ndarray,
    *,
    min_luminance_std: float = 12.0,
    min_luminance_mean: float = 18.0,
    min_luminance_range: float = 25.0,
) -> bool:
    """Return False for near-black or flat frames with no usable structure."""
    metrics = image_quality_metrics(image)
    return (
        metrics.luminance_std >= min_luminance_std
        and metrics.luminance_mean >= min_luminance_mean
        and metrics.luminance_range >= min_luminance_range
    )


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1).astype(np.float32)
        return image.astype(np.float32)
    raise TypeError(f"Unsupported image type: {type(image)!r}")
