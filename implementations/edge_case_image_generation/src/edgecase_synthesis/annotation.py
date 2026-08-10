"""Open-vocabulary detection via YOLO-World (boxes for detector training / judge)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from edgecase_synthesis.conditioning import resolve_device


@dataclass(frozen=True)
class Detection:
    """One labeled region in an annotated image."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray  # bool (H, W); box-filled placeholder for viz compat


@dataclass
class AnnotationResult:
    """Detections for one image (boxes; masks are box fills only)."""

    detections: list[Detection]
    overlay: np.ndarray  # uint8 RGB
    num_instances: int


class OpenVocabAnnotator:
    """YOLO-World open-vocab boxes (stand-in for Grounded-SAM-2 labeling).

    Produces class labels + xyxy boxes for training a detector and for the
    VLM judge summary. No SAM masks — our eval target is YOLOv8-style detection.
    """

    def __init__(
        self,
        *,
        detector_model: str = "yolov8s-worldv2.pt",
        classes: list[str] | None = None,
        device: str | None = None,
        conf: float = 0.15,
        max_detections: int = 20,
        sam_model: str | None = None,  # unused; kept for config back-compat
    ) -> None:
        del sam_model  # explicitly unused
        self.detector_model = detector_model
        self.classes = list(classes or [])
        if not self.classes:
            raise ValueError(
                "annotation.classes is empty — set classes in "
                "configs/datasets/<dataset>/annotation.yaml"
            )
        self.conf = float(conf)
        self.max_detections = int(max_detections)
        self.device = resolve_device(device)
        weights = _resolve_yolo_weights(detector_model)
        self.model = YOLO(weights)
        # Keep detector + CLIP on CPU until predict(); set_classes tokenizes on CPU
        # and will crash if CLIP weights were already placed on CUDA.
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        self._yolo_device = "cuda:0" if self.device.type == "cuda" else "cpu"
        self._active_classes: list[str] | None = None

    @torch.inference_mode()
    def annotate(
        self,
        image: Image.Image | np.ndarray,
        *,
        classes: list[str] | None = None,
        conf: float | None = None,
        overlay_alpha: float = 0.35,
        seed_mask: np.ndarray | None = None,
        seed_label: str | None = None,
        seed_confidence: float = 0.99,
        require_seed_if_empty: bool = True,
        target_labels: set[str] | None = None,
    ) -> AnnotationResult:
        """Run YOLO-World with the given class vocabulary.

        If open-vocab returns no **target** boxes and ``seed_mask`` / ``seed_label``
        are provided, add one box from the edit region (pseudo-label for training).
        """
        pil_image = _to_pil(image)
        rgb = np.array(pil_image)
        active_classes = [str(c).strip() for c in (classes or self.classes) if str(c).strip()]
        if not active_classes:
            raise ValueError("No annotation classes provided")
        threshold = self.conf if conf is None else float(conf)

        self._set_classes(active_classes)
        results = self.model.predict(
            source=rgb,
            conf=threshold,
            verbose=False,
            device=self._yolo_device,
            max_det=self.max_detections,
        )
        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                names = results[0].names or {}
                order = np.argsort(-scores)[: self.max_detections]
                for idx in order:
                    cid = int(cls_ids[idx])
                    if cid in names:
                        label = str(names[cid])
                    elif 0 <= cid < len(active_classes):
                        label = active_classes[cid]
                    else:
                        label = str(cid)
                    x1, y1, x2, y2 = (int(v) for v in xyxy[idx])
                    mask = _box_mask(rgb.shape[:2], (x1, y1, x2, y2))
                    detections.append(
                        Detection(
                            label=label,
                            confidence=float(scores[idx]),
                            bbox_xyxy=(x1, y1, x2, y2),
                            mask=mask,
                        )
                    )

        tmp = AnnotationResult(
            detections=detections,
            overlay=rgb,
            num_instances=len(detections),
        )
        has_target = (
            has_target_detections(tmp, target_labels)
            if target_labels
            else bool(detections)
        )
        if (
            require_seed_if_empty
            and seed_mask is not None
            and seed_label
            and not has_target
        ):
            seeded = detection_from_mask(
                seed_mask,
                label=str(seed_label),
                confidence=float(seed_confidence),
                image_hw=rgb.shape[:2],
            )
            if seeded is not None:
                detections.append(seeded)

        overlay = _draw_overlay(rgb, detections, alpha=overlay_alpha)
        return AnnotationResult(
            detections=detections,
            overlay=overlay,
            num_instances=len(detections),
        )

    def _set_classes(self, active_classes: list[str]) -> None:
        """Encode class prompts on CPU to avoid CLIP token/weight device mismatch.

        Ultralytics YOLO-World keeps tokenized text on CPU while CLIP embeddings
        may already live on CUDA after a prior ``predict`` — that raises
        ``Expected all tensors to be on the same device``.
        """
        if self._active_classes == active_classes:
            return
        # CLIP encode must see matching devices; safest is CPU for set_classes.
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001 — some ultralytics builds lack .to
            pass
        if hasattr(self.model, "clip_model") and self.model.clip_model is not None:
            try:
                self.model.clip_model.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        self.model.set_classes(active_classes)
        self._active_classes = list(active_classes)

    def unload(self) -> None:
        """Drop model weights to free VRAM before loading the judge."""
        self.model = None  # type: ignore[assignment]
        self._active_classes = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | Any, device: str | None = None):
        annotation = cfg.get("annotation", cfg)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            detector_model=str(
                annotation.get("detector_model", "yolov8s-worldv2.pt")
            ),
            classes=list(annotation.get("classes", [])),
            conf=float(annotation.get("conf", 0.15)),
            max_detections=int(annotation.get("max_detections", 20)),
            device=device,
            sam_model=annotation.get("sam_model"),
        )


def _resolve_yolo_weights(detector_model: str) -> str:
    path = Path(detector_model)
    if path.exists():
        return str(path.resolve())
    name = path.name if path.suffix else detector_model
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / name
        if candidate.exists():
            return str(candidate.resolve())
        if name != "yolov8s-worldv2.pt":
            fallback = parent / "yolov8s-worldv2.pt"
            if fallback.exists():
                return str(fallback.resolve())
    return detector_model  # let ultralytics download / error clearly


# Scene-context YOLO queries — not valid "target object" evidence for accept gates.
_CONTEXT_LABELS = {
    "road",
    "asphalt",
    "car",
    "person",
    "sidewalk",
    "building",
    "sky",
    "tree",
}


def canonicalize_label(label: str) -> str:
    return str(label).lower().strip().replace("-", " ").replace("_", " ")


def target_label_names(
    anomaly_id: str,
    annotation_classes: list[str] | None = None,
) -> set[str]:
    """Labels that count as the rare object (excludes road/car context queries)."""
    names = {
        canonicalize_label(anomaly_id),
        canonicalize_label(anomaly_id.replace("_", " ")),
    }
    for raw in annotation_classes or []:
        canon = canonicalize_label(raw)
        if canon and canon not in _CONTEXT_LABELS:
            names.add(canon)
    return names


def has_target_detections(
    annotation: AnnotationResult | None,
    target_labels: set[str],
) -> bool:
    if annotation is None or not target_labels:
        return False
    for det in annotation.detections:
        if canonicalize_label(det.label) in target_labels:
            return True
    return False


def detection_from_mask(
    mask: np.ndarray,
    *,
    label: str,
    confidence: float = 0.99,
    image_hw: tuple[int, int] | None = None,
) -> Detection | None:
    """Axis-aligned box from a boolean / {0,1} edit mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.any(axis=-1)
    if image_hw is not None and arr.shape[:2] != image_hw:
        # Nearest resize to the annotated image (generation may have resized).
        import cv2

        h, w = image_hw
        arr = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    binary = arr.astype(bool)
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    box = (x1, y1, x2, y2)
    hw = image_hw or binary.shape[:2]
    return Detection(
        label=str(label),
        confidence=float(confidence),
        bbox_xyxy=box,
        mask=_box_mask(hw, box),
    )


def _box_mask(hw: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = hw
    x1, y1, x2, y2 = box
    m = np.zeros((h, w), dtype=bool)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _draw_overlay(
    rgb: np.ndarray,
    detections: list[Detection],
    *,
    alpha: float,
) -> np.ndarray:
    overlay = rgb.copy()
    rng = np.random.default_rng(7)

    for det in detections:
        color = tuple(int(v) for v in rng.integers(40, 255, size=3))
        mask_rgb = np.zeros_like(rgb)
        mask_rgb[det.mask] = color
        overlay = np.clip(
            overlay.astype(np.float32) * (1 - alpha) + mask_rgb * alpha,
            0,
            255,
        ).astype(np.uint8)

        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay
