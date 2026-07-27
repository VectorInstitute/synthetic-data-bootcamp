"""Load real images (and optional detection labels) from Hydra data sources."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from omegaconf import DictConfig, OmegaConf
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USER_AGENT = "edgecase-synthesis/0.2 (educational)"


@dataclass(frozen=True)
class ImageSample:
    """A single real image ready for the pipeline."""

    path: Path
    image: Image.Image
    name: str

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass(frozen=True)
class DetectionBox:
    """One ground-truth box (xyxy pixels) with a class name."""

    label: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DataSourceInfo:
    name: str
    label: str
    license: str
    attribution: str


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml). "
        "Run from the implementation folder or set PROJECT_ROOT."
    )


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def get_data_source_info(cfg: DictConfig | dict[str, Any]) -> DataSourceInfo:
    data = _as_dict(cfg)["data"]
    return DataSourceInfo(
        name=data["name"],
        label=data.get("label", data["name"]),
        license=data.get("license", ""),
        attribution=data.get("attribution", ""),
    )


def list_sample_images(samples_dir: Path | str) -> list[Path]:
    root = Path(samples_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_sample_images(
    samples_dir: Path | str | None = None,
    *,
    cfg: DictConfig | dict[str, Any] | None = None,
) -> list[ImageSample]:
    if cfg is None and samples_dir is None:
        from edgecase_synthesis.config import load_config

        cfg = load_config()
    config = _as_dict(cfg) if cfg is not None else {}
    root = Path(samples_dir or config["paths"]["samples_dir"])
    max_images = int(config.get("data", {}).get("max_images", 0) or 0)
    samples: list[ImageSample] = []
    for path in list_sample_images(root):
        image = Image.open(path).convert("RGB")
        samples.append(ImageSample(path=path, image=image, name=path.stem))
        if max_images and len(samples) >= max_images:
            break
    return samples


def load_detection_labels(
    samples_dir: Path | str | None = None,
    *,
    cfg: DictConfig | dict[str, Any] | None = None,
) -> dict[str, list[DetectionBox]]:
    """Load optional bbox labels written next to samples (labels.json)."""
    if cfg is not None and samples_dir is None:
        samples_dir = _as_dict(cfg)["paths"]["samples_dir"]
    path = Path(samples_dir) / "labels.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[DetectionBox]] = {}
    for name, boxes in raw.items():
        out[name] = [
            DetectionBox(
                label=str(b["label"]),
                bbox_xyxy=tuple(float(x) for x in b["bbox_xyxy"]),  # type: ignore[arg-type]
            )
            for b in boxes
        ]
    return out


def prepare_sample_images(
    samples_dir: Path | None = None,
    *,
    cfg: DictConfig | dict[str, Any] | None = None,
    force: bool = False,
) -> list[Path]:
    """Populate samples_dir from the active Hydra data source."""
    if cfg is None:
        from edgecase_synthesis.config import load_config

        cfg = load_config()

    config = _as_dict(cfg)
    target = (samples_dir or Path(config["paths"]["samples_dir"])).resolve()
    target.mkdir(parents=True, exist_ok=True)
    source = config["data"]
    kind = str(source.get("kind", source.get("name", "local"))).lower()

    if not force and list_sample_images(target):
        # Refresh labels.json if missing but source can provide it.
        if kind in {"hf_detection", "neu_det"} and not (target / "labels.json").exists():
            pass  # fall through to rebuild
        else:
            return list_sample_images(target)

    if kind == "local":
        existing = list_sample_images(target)
        if existing:
            return existing
        raise FileNotFoundError(
            f"No images in {target}. Drop files there or switch "
            "data/source@data=neu_det|urls|nordland_hf."
        )
    if kind == "urls":
        return _prepare_urls(target, source, force=force)
    if kind in {"hf_detection", "neu_det"}:
        return _prepare_hf_detection(target, source, force=force)
    if kind == "hf_rows":
        return _prepare_hf_rows(target, source, force=force)
    raise ValueError(
        f"Unknown data.kind={kind!r}. Use local | urls | hf_detection | hf_rows."
    )


def _prepare_urls(target: Path, source: dict[str, Any], *, force: bool) -> list[Path]:
    images = dict(source.get("images") or {})
    if not images:
        raise ValueError("urls source needs data.images: {filename: url}")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    saved: list[Path] = []
    for name, url in images.items():
        dest = target / name
        if dest.exists() and not force:
            saved.append(dest)
            continue
        response = session.get(str(url), timeout=60)
        response.raise_for_status()
        Image.open(BytesIO(response.content)).convert("RGB").save(dest)
        saved.append(dest)
    return saved


def _prepare_hf_detection(target: Path, source: dict[str, Any], *, force: bool) -> list[Path]:
    """Pull a detection dataset via `datasets` and cache RGB + labels.json."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install `datasets` to use hf_detection sources: uv add datasets"
        ) from exc

    hf_id = str(source.get("hf_id", ""))
    if not hf_id:
        raise ValueError("hf_detection source requires data.hf_id")
    split = str(source.get("hf_split", "train"))
    max_images = int(source.get("max_images", 8))
    prefer = [str(c).lower() for c in (source.get("prefer_classes") or [])]

    ds = load_dataset(hf_id, split=split, streaming=True)
    labels_out: dict[str, list[dict[str, Any]]] = {}
    saved: list[Path] = []
    class_counts: dict[str, int] = {}

    for idx, row in enumerate(ds):
        if len(saved) >= max_images:
            break
        image, boxes = _extract_hf_detection_row(row)
        if image is None:
            continue
        primary = boxes[0].label.lower() if boxes else ""
        if prefer:
            # Round-robin prefer_classes when possible.
            want = prefer[len(saved) % len(prefer)]
            labels_lower = {b.label.lower() for b in boxes}
            if want not in labels_lower and primary and primary not in prefer:
                # Soft skip early rows that don't help class coverage.
                if idx < max_images * 20 and len(saved) < max_images:
                    continue
        stem = f"neu_{len(saved):04d}"
        if primary:
            stem = f"{primary.replace(' ', '_')}_{len(saved):04d}"
            class_counts[primary] = class_counts.get(primary, 0) + 1
        dest = target / f"{stem}.png"
        if dest.exists() and not force:
            saved.append(dest)
        else:
            image.convert("RGB").save(dest)
            saved.append(dest)
        labels_out[stem] = [
            {"label": b.label, "bbox_xyxy": list(b.bbox_xyxy)} for b in boxes
        ]

    if not saved:
        raise RuntimeError(
            f"No images extracted from {hf_id}. Check schema or use kind: local."
        )
    (target / "labels.json").write_text(json.dumps(labels_out, indent=2), encoding="utf-8")
    meta = {
        "hf_id": hf_id,
        "split": split,
        "classes": source.get("classes", []),
        "class_counts": class_counts,
    }
    (target / "source_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return saved


def _extract_hf_detection_row(row: dict[str, Any]) -> tuple[Image.Image | None, list[DetectionBox]]:
    """Best-effort parse of common HF detection schemas (incl. AI4Manufacturing/191)."""
    image = row.get("image") or row.get("img")
    if image is None and "images" in row and row["images"]:
        image = row["images"][0]
    if image is not None and not isinstance(image, Image.Image):
        try:
            image = Image.open(BytesIO(image["bytes"])).convert("RGB")
        except Exception:  # noqa: BLE001
            image = None
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    boxes: list[DetectionBox] = []
    meta = row.get("metadata") or {}
    objects = meta.get("objects") or row.get("objects") or {}
    # COCO-ish: objects = {bbox: [...], category: [...]}
    if isinstance(objects, dict) and "bbox" in objects:
        cats = objects.get("category") or objects.get("categories") or objects.get("label") or []
        for bbox, cat in zip(objects["bbox"], cats, strict=False):
            label = str(cat)
            if isinstance(cat, int) and "categories" in meta:
                label = str(meta["categories"][cat])
            boxes.append(_box_from_any(bbox, label, image))
    elif isinstance(objects, list):
        for obj in objects:
            label = str(obj.get("label") or obj.get("category") or "defect")
            bbox = obj.get("bbox") or obj.get("bbox_xyxy") or obj.get("box")
            if bbox is not None:
                boxes.append(_box_from_any(bbox, label, image))

    # AI4Manufacturing annot lines: "class,[x,y,w,h]"
    annot = row.get("annot") or row.get("answer")
    if isinstance(annot, str) and image is not None:
        for line in annot.strip().splitlines():
            line = line.strip()
            if not line or "," not in line:
                continue
            label, rest = line.split(",", 1)
            nums = [float(x) for x in rest.strip("[] ").replace(",", " ").split() if x]
            if len(nums) >= 4:
                boxes.append(_box_from_any(nums[:4], label.strip(), image))

    return image, boxes


def _box_from_any(
    bbox: Any,
    label: str,
    image: Image.Image | None,
) -> DetectionBox:
    vals = [float(x) for x in list(bbox)[:4]]
    # Heuristic: COCO xywh if w,h look like sizes; else assume xyxy.
    if image is not None and vals[2] < image.size[0] and vals[3] < image.size[1]:
        # If x2,y2 would exceed image when treated as xywh→xyxy expansion needed.
        if vals[0] + vals[2] <= image.size[0] * 1.05 and vals[1] + vals[3] <= image.size[1] * 1.05:
            # Treat as xywh when fourth value is small relative to height.
            if vals[2] < image.size[0] * 0.95 and vals[3] < image.size[1] * 0.95:
                x1, y1, w, h = vals
                return DetectionBox(label=label, bbox_xyxy=(x1, y1, x1 + w, y1 + h))
    x1, y1, x2, y2 = vals
    return DetectionBox(label=label, bbox_xyxy=(x1, y1, x2, y2))


def _prepare_hf_rows(target: Path, source: dict[str, Any], *, force: bool) -> list[Path]:
    """Fetch individual rows from HF datasets-server (image-only sources)."""
    frames = list(source.get("frames") or [])
    if not frames:
        raise ValueError("hf_rows source needs data.frames: [{offset, save_as}, ...]")
    hf_id = str(source["hf_id"])
    split = str(source.get("split", "train"))
    config_name = source.get("config")
    api = str(source.get("rows_api", "https://datasets-server.huggingface.co/rows"))
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    saved: list[Path] = []
    for frame in frames:
        dest = target / str(frame["save_as"])
        if dest.exists() and not force:
            saved.append(dest)
            continue
        params = {"dataset": hf_id, "split": split, "offset": int(frame["offset"]), "length": 1}
        if config_name:
            params["config"] = config_name
        url = f"{api}?{urlencode(params)}"
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                response = session.get(url, timeout=60)
                response.raise_for_status()
                row = response.json()["rows"][0]["row"]
                image_info = row.get("image") or {}
                # Prefer bytes via nested fetch if URL present.
                img_url = image_info.get("src") or image_info.get("url")
                if img_url:
                    img_resp = session.get(img_url, timeout=60)
                    img_resp.raise_for_status()
                    Image.open(BytesIO(img_resp.content)).convert("RGB").save(dest)
                else:
                    raise KeyError("No image URL in HF row")
                saved.append(dest)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(f"Failed to fetch HF row {frame}: {last_err}") from last_err
    return saved
