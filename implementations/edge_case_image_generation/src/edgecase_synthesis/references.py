"""Auto-pick real same-class reference images for post-gen fidelity judging."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from edgecase_synthesis.data import DetectionBox
from edgecase_synthesis.eda import group_by_tag, list_tagged_images, load_labels_for_dir


@dataclass
class ReferenceImage:
    """One real reference shown to the VLM judge."""

    image: Image.Image
    path: Path
    role: str  # full | crop
    label: str = ""


def pick_class_references(
    samples_dir: Path | str,
    class_id: str,
    *,
    n_full: int = 2,
    n_crops: int = 1,
    seed: int = 42,
    exclude_stems: set[str] | None = None,
    labels: dict[str, list[DetectionBox]] | None = None,
    prefixes: list[str] | None = None,
    max_side: int = 1024,
) -> list[ReferenceImage]:
    """Pick real same-class photos (+ optional GT crops) for fidelity checks.

    Prefers filename-tagged extract buckets (``traffic_cone_*``, ``trash_bin_*``).
    Crops use ``labels.json`` boxes for that class when available.
    """
    samples_dir = Path(samples_dir)
    class_id = str(class_id).strip()
    if not class_id or (n_full <= 0 and n_crops <= 0):
        return []

    paths = list_tagged_images(samples_dir)
    by_tag = group_by_tag(paths, tags=[class_id], prefixes=prefixes)
    pool = list(by_tag.get(class_id) or [])
    exclude = {str(s) for s in (exclude_stems or set())}
    pool = [p for p in pool if p.stem not in exclude and p.name not in exclude]
    if not pool:
        return []

    rng = random.Random(int(seed) + sum(ord(c) for c in class_id))
    rng.shuffle(pool)

    if labels is None:
        try:
            labels = load_labels_for_dir(samples_dir)
        except Exception:
            labels = {}

    out: list[ReferenceImage] = []
    # Full-frame class photos (global / scene fidelity).
    for path in pool:
        if len([r for r in out if r.role == "full"]) >= int(n_full):
            break
        try:
            img = _load_rgb(path, max_side=max_side)
        except Exception:
            continue
        out.append(ReferenceImage(image=img, path=path, role="full", label=class_id))

    # GT crops of the rare object (object fidelity).
    if int(n_crops) > 0 and labels:
        crop_count = 0
        for path in pool:
            if crop_count >= int(n_crops):
                break
            boxes = _boxes_for_path(path, labels)
            class_boxes = [b for b in boxes if _label_matches(b.label, class_id)]
            if not class_boxes:
                continue
            # Prefer a mid-sized box (not tiny noise).
            class_boxes.sort(key=_box_area, reverse=True)
            box = class_boxes[0]
            try:
                full = Image.open(path).convert("RGB")
                crop = _crop_box(full, box.bbox_xyxy, pad=0.08, max_side=max_side)
            except Exception:
                continue
            out.append(
                ReferenceImage(image=crop, path=path, role="crop", label=str(box.label))
            )
            crop_count += 1

    return out


def _load_rgb(path: Path, *, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_side > 0 and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale)))),
            Image.Resampling.LANCZOS,
        )
    return img


def _boxes_for_path(
    path: Path, labels: dict[str, list[DetectionBox]]
) -> list[DetectionBox]:
    return list(labels.get(path.name) or labels.get(path.stem) or [])


def _label_matches(label: str, class_id: str) -> bool:
    a = str(label).lower().replace("-", " ").replace("_", " ").strip()
    b = str(class_id).lower().replace("-", " ").replace("_", " ").strip()
    return a == b or b in a or a in b


def _box_area(box: DetectionBox) -> float:
    x1, y1, x2, y2 = (float(v) for v in box.bbox_xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _crop_box(
    image: Image.Image,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    pad: float = 0.08,
    max_side: int = 1024,
) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0.0, x1 - pad * bw)
    y1 = max(0.0, y1 - pad * bh)
    x2 = min(float(w), x2 + pad * bw)
    y2 = min(float(h), y2 + pad * bh)
    crop = image.crop((int(x1), int(y1), int(max(x1 + 1, x2)), int(max(y1 + 1, y2))))
    if max_side > 0 and max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (
                max(1, int(round(crop.size[0] * scale))),
                max(1, int(round(crop.size[1] * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    return crop
