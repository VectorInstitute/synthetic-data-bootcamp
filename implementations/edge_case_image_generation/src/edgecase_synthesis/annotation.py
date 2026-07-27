"""Open-vocabulary detection + SAM masks — local stand-in for Grounded-SAM 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from ultralytics import SAM

from edgecase_synthesis.conditioning import resolve_device


@dataclass(frozen=True)
class Detection:
    """One labeled region in an annotated image."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray  # bool, shape (H, W)


@dataclass
class AnnotationResult:
    """Detections and masks for one image."""

    detections: list[Detection]
    overlay: np.ndarray  # uint8 RGB
    num_instances: int


# OWL-ViT was trained with CLIP-style captions. Bare nouns ("sky") score ~0.02;
# "a photo of …" lifts scores enough to clear a usable threshold.
_VOWELS = set("aeiou")


def _owlvit_query(label: str) -> str:
    text = label.strip()
    lower = text.lower()
    if lower.startswith("a photo of"):
        return text
    article = "an" if lower[:1] in _VOWELS else "a"
    return f"a photo of {article} {text}"


def _display_label(query_or_label: str) -> str:
    text = query_or_label.strip()
    lower = text.lower()
    for prefix in ("a photo of an ", "a photo of a ", "a photo of the ", "a photo of "):
        if lower.startswith(prefix):
            return text[len(prefix) :]
    return text


class OpenVocabAnnotator:
    """OWL-ViT detections refined with MobileSAM box prompts.

    Grounded-SAM 2 would replace this stack in production for tighter
    text-to-mask alignment; OWL-ViT + MobileSAM runs locally without extra
    CLIP installs and is much lighter than the full G-SAM 2 pipeline.
    """

    def __init__(
        self,
        *,
        detector_model: str = "google/owlvit-base-patch32",
        sam_model: str = "mobile_sam.pt",
        classes: list[str] | None = None,
        device: str | None = None,
        conf: float = 0.015,
        max_detections: int = 20,
    ) -> None:
        self.detector_model = detector_model
        self.sam_model = sam_model
        self.classes = classes or [
            "pothole",
            "road crack",
            "traffic cone",
            "road",
        ]
        self.conf = float(conf)
        self.max_detections = int(max_detections)
        self.device = resolve_device(device)
        self.processor = OwlViTProcessor.from_pretrained(detector_model)
        self.detector = OwlViTForObjectDetection.from_pretrained(detector_model)
        self.detector.to(self.device).eval()
        self.sam = SAM(_resolve_sam_path(sam_model))

    @torch.inference_mode()
    def annotate(
        self,
        image: Image.Image | np.ndarray,
        *,
        classes: list[str] | None = None,
        conf: float | None = None,
        overlay_alpha: float = 0.45,
        seed_mask: np.ndarray | None = None,
        seed_label: str | None = None,
        seed_confidence: float = 0.99,
    ) -> AnnotationResult:
        """Detect open-vocab boxes, refine with SAM, optionally seed a known mask.

        ``seed_mask`` / ``seed_label`` are for synthetic inserts when the
        open-vocab detector misses the edited region: we inject the edit mask
        as a high-confidence detection for the anomaly class.
        """
        pil_image = _to_pil(image)
        rgb = np.array(pil_image)
        active_classes = list(classes or self.classes)
        threshold = self.conf if conf is None else float(conf)

        queries = [_owlvit_query(c) for c in active_classes]
        inputs = self.processor(
            text=[queries],
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.detector(**inputs)

        target_size = torch.tensor([pil_image.size[::-1]], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_size,
            threshold=threshold,
            text_labels=[queries],
        )[0]

        detections: list[Detection] = []

        if len(processed["scores"]) > 0:
            boxes_xyxy = processed["boxes"].cpu().numpy()
            confidences = processed["scores"].cpu().numpy()
            text_labels = processed.get("text_labels")
            if text_labels is None:
                label_ids = processed["labels"].cpu().numpy().astype(int)
                text_labels = [queries[i] for i in label_ids]

            # Keep the strongest boxes (OWL-ViT is noisy at low thresholds).
            order = np.argsort(-confidences)[: self.max_detections]
            boxes_xyxy = boxes_xyxy[order]
            confidences = confidences[order]
            text_labels = [text_labels[i] for i in order]

            # Soft NMS by IoU to reduce duplicate "tree" tiles.
            keep = _nms_indices(boxes_xyxy, confidences, iou_threshold=0.5)
            boxes_xyxy = boxes_xyxy[keep]
            confidences = confidences[keep]
            text_labels = [text_labels[i] for i in keep]

            masks = _sam_masks_for_boxes(self.sam, rgb, boxes_xyxy)
            for idx, (box, score, query) in enumerate(
                zip(boxes_xyxy, confidences, text_labels, strict=True)
            ):
                x1, y1, x2, y2 = (int(v) for v in box)
                mask = masks[idx] if idx < len(masks) else np.zeros(rgb.shape[:2], bool)
                detections.append(
                    Detection(
                        label=_display_label(str(query)),
                        confidence=float(score),
                        bbox_xyxy=(x1, y1, x2, y2),
                        mask=mask,
                    )
                )

        if seed_mask is not None and seed_label:
            seed = _detection_from_mask(
                seed_mask,
                label=seed_label,
                confidence=seed_confidence,
                image_hw=rgb.shape[:2],
            )
            if seed is not None:
                detections = [seed, *detections]

        overlay = _draw_overlay(rgb, detections, alpha=overlay_alpha)
        return AnnotationResult(
            detections=detections,
            overlay=overlay,
            num_instances=len(detections),
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | Any, device: str | None = None):
        annotation = cfg.get("annotation", cfg)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            detector_model=annotation.get(
                "detector_model", "google/owlvit-base-patch32"
            ),
            sam_model=annotation.get("sam_model", "mobile_sam.pt"),
            classes=list(annotation.get("classes", [])),
            conf=float(annotation.get("conf", 0.015)),
            max_detections=int(annotation.get("max_detections", 20)),
            device=device,
        )


def _resolve_sam_path(sam_model: str) -> str:
    path = Path(sam_model)
    if path.exists():
        return str(path)
    # Try project root / CWD parents (notebooks often run from notebooks/).
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / sam_model
        if candidate.exists():
            return str(candidate)
        candidate = parent / "mobile_sam.pt"
        if candidate.exists():
            return str(candidate)
    return sam_model  # let ultralytics download / error clearly


def _sam_masks_for_boxes(sam: SAM, rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros((0, *rgb.shape[:2]), dtype=bool)
    sam_results = sam.predict(source=rgb, bboxes=boxes_xyxy, verbose=False)
    sam_out = sam_results[0]
    if sam_out.masks is None:
        return np.zeros((len(boxes_xyxy), *rgb.shape[:2]), dtype=bool)
    return sam_out.masks.data.cpu().numpy().astype(bool)


def _detection_from_mask(
    mask: np.ndarray,
    *,
    label: str,
    confidence: float,
    image_hw: tuple[int, int],
) -> Detection | None:
    h, w = image_hw
    m = mask.astype(bool)
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return Detection(
        label=label,
        confidence=float(confidence),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        mask=m,
    )


def _nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float,
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


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
