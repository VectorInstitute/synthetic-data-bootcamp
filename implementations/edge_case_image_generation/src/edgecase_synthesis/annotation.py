"""Open-vocabulary detection + SAM masks — local stand-in for Grounded-SAM 2."""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        self.detector_model = detector_model
        self.sam_model = sam_model
        self.classes = classes or [
            "railway track",
            "train",
            "sky",
            "tree",
            "traffic cone",
            "fallen tree",
            "deer",
            "dog",
        ]
        self.device = resolve_device(device)
        self.processor = OwlViTProcessor.from_pretrained(detector_model)
        self.detector = OwlViTForObjectDetection.from_pretrained(detector_model)
        self.detector.to(self.device).eval()
        self.sam = SAM(sam_model)

    @torch.inference_mode()
    def annotate(
        self,
        image: Image.Image | np.ndarray,
        *,
        classes: list[str] | None = None,
        conf: float = 0.08,
        overlay_alpha: float = 0.45,
    ) -> AnnotationResult:
        pil_image = _to_pil(image)
        rgb = np.array(pil_image)
        active_classes = classes or self.classes

        inputs = self.processor(
            text=[active_classes],
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.detector(**inputs)

        target_size = torch.tensor([pil_image.size[::-1]], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_size,
            threshold=conf,
            text_labels=[active_classes],
        )[0]

        if len(processed["scores"]) == 0:
            return AnnotationResult(
                detections=[],
                overlay=rgb.copy(),
                num_instances=0,
            )

        boxes_xyxy = processed["boxes"].cpu().numpy()
        confidences = processed["scores"].cpu().numpy()
        label_ids = processed["labels"].cpu().numpy().astype(int)
        labels = [active_classes[label_id] for label_id in label_ids]

        sam_results = self.sam.predict(
            source=rgb,
            bboxes=boxes_xyxy,
            verbose=False,
        )
        sam = sam_results[0]
        masks = (
            sam.masks.data.cpu().numpy().astype(bool)
            if sam.masks is not None
            else np.zeros((len(boxes_xyxy), *rgb.shape[:2]), dtype=bool)
        )

        detections: list[Detection] = []
        for idx, (box, score, label) in enumerate(
            zip(boxes_xyxy, confidences, labels, strict=True)
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            mask = masks[idx] if idx < len(masks) else np.zeros(rgb.shape[:2], bool)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask=mask,
                )
            )

        overlay = _draw_overlay(rgb, detections, alpha=overlay_alpha)
        return AnnotationResult(
            detections=detections,
            overlay=overlay,
            num_instances=len(detections),
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | Any, device: str | None = None):
        annotation = cfg.get("annotation", cfg)
        return cls(
            detector_model=annotation.get(
                "detector_model", "google/owlvit-base-patch32"
            ),
            sam_model=annotation.get("sam_model", "mobile_sam.pt"),
            classes=list(annotation.get("classes", [])),
            device=device,
        )


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
