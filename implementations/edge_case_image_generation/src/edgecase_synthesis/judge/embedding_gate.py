"""Slice C — CLIP embedding safe-zone gates (fidelity + novelty).

Fidelity: candidate must be close enough to the nearest **real** same-class
image (global full-frame and optional local crop).

Novelty: candidate must not be too close to the nearest neighbor in
**real ∪ already-accepted synth** (same embedding space).

Bootcamp scale (~tens–hundreds per class) → brute-force cosine; no FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image

from edgecase_synthesis.generate.conditioning import resolve_device
from edgecase_synthesis.data.loader import DetectionBox
from edgecase_synthesis.data.eda import group_by_tag, list_tagged_images, load_labels_for_dir
from edgecase_synthesis.judge.references import _crop_box, _label_matches


@dataclass
class EmbeddingGateConfig:
    enabled: bool = True
    model_id: str = "openai/clip-vit-base-patch32"
    # Cosine similarity in [−1, 1]; CLIP image pairs are typically ~0.5–0.95.
    min_real_sim_global: float = 0.58
    min_real_sim_local: float = 0.52
    max_neighbor_sim: float = 0.94
    use_local: bool = True
    fail_action: str = "retry"  # retry | reject
    max_bank: int = 256
    max_side: int = 224


@dataclass
class EmbeddingGateMetrics:
    """Per-candidate embedding knn scores (logged on judge / manifest)."""

    enabled: bool = False
    model_id: str = ""
    real_sim_global: float | None = None
    real_sim_local: float | None = None
    neighbor_sim: float | None = None
    nearest_real: str | None = None
    nearest_neighbor: str | None = None
    failed_fidelity: bool = False
    failed_novelty: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_id": self.model_id,
            "real_sim_global": self.real_sim_global,
            "real_sim_local": self.real_sim_local,
            "neighbor_sim": self.neighbor_sim,
            "nearest_real": self.nearest_real,
            "nearest_neighbor": self.nearest_neighbor,
            "failed_fidelity": self.failed_fidelity,
            "failed_novelty": self.failed_novelty,
            "reason": self.reason,
        }


@dataclass
class _BankEntry:
    vector: np.ndarray
    path: str
    kind: str  # real | synth
    role: str  # full | crop


@dataclass
class _ClassBanks:
    real_full: list[_BankEntry] = field(default_factory=list)
    real_crop: list[_BankEntry] = field(default_factory=list)
    neighbor_full: list[_BankEntry] = field(default_factory=list)  # real ∪ synth


class ClipImageEncoder:
    """Thin CLIP image encoder (shared with judge fallback weights)."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        *,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self._processor = None
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._model.to(self.device).eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalized 1-D CLIP image embedding."""
        self._ensure()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Prefer the documented pooled path; some transformers builds have returned
        # token sequences from get_image_features, which breaks cosine KNN.
        vision_out = self._model.vision_model(
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )
        pooled = vision_out.pooler_output
        if pooled is None:
            # Fall back to CLS token if pooler is missing.
            pooled = vision_out.last_hidden_state[:, 0]
        feats = self._model.visual_projection(pooled)
        vec = feats.detach().float().cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec) + 1e-8)
        return (vec / norm).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def nearest_neighbor(
    query: np.ndarray, bank: list[_BankEntry]
) -> tuple[float | None, str | None]:
    if not bank:
        return None, None
    best_sim = -1.0
    best_path = None
    for entry in bank:
        sim = cosine_sim(query, entry.vector)
        if sim > best_sim:
            best_sim = sim
            best_path = entry.path
    return best_sim, best_path


def crop_from_mask_or_boxes(
    image: Image.Image,
    *,
    edit_mask: np.ndarray | None = None,
    boxes: list[Any] | None = None,
    target_labels: list[str] | None = None,
    max_side: int = 224,
) -> Image.Image | None:
    """Build a local crop from edit mask or target detection boxes."""
    w, h = image.size
    if boxes:
        targets = []
        for box in boxes:
            label = str(getattr(box, "label", "") or "")
            if target_labels and not any(_label_matches(label, t) for t in target_labels):
                continue
            bbox = getattr(box, "bbox_xyxy", None)
            if bbox is None:
                continue
            targets.append(bbox)
        if targets:
            # Largest target box.
            targets.sort(
                key=lambda b: max(0.0, float(b[2]) - float(b[0]))
                * max(0.0, float(b[3]) - float(b[1])),
                reverse=True,
            )
            return _crop_box(image, targets[0], pad=0.08, max_side=max_side)

    if edit_mask is not None:
        mask = np.asarray(edit_mask).astype(bool)
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        if mask.shape[:2] != (h, w):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        ys, xs = np.where(mask)
        if len(xs) >= 8:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            return _crop_box(image, (x1, y1, x2, y2), pad=0.08, max_side=max_side)
    return None


class EmbeddingGate:
    """Per-class CLIP banks + safe-zone evaluation."""

    def __init__(
        self,
        config: EmbeddingGateConfig | None = None,
        *,
        samples_dir: Path | str | None = None,
        stem_prefixes: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or EmbeddingGateConfig()
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self.stem_prefixes = list(stem_prefixes) if stem_prefixes else None
        self.encoder = ClipImageEncoder(self.config.model_id, device=device)
        self._banks: dict[str, _ClassBanks] = {}
        self._labels: dict[str, list[DetectionBox]] | None = None
        self._lock = Lock()

    def unload(self) -> None:
        self.encoder.unload()

    def _load_labels(self) -> dict[str, list[DetectionBox]]:
        if self._labels is not None:
            return self._labels
        if self.samples_dir is None:
            self._labels = {}
            return self._labels
        try:
            self._labels = load_labels_for_dir(self.samples_dir)
        except Exception:
            self._labels = {}
        return self._labels

    def _ensure_real_bank(self, class_id: str) -> _ClassBanks:
        class_id = str(class_id)
        banks = self._banks.setdefault(class_id, _ClassBanks())
        if banks.real_full:
            return banks
        if self.samples_dir is None:
            return banks

        paths = list_tagged_images(self.samples_dir)
        by_tag = group_by_tag(paths, tags=[class_id], prefixes=self.stem_prefixes)
        pool = list(by_tag.get(class_id) or [])[: int(self.config.max_bank)]
        labels = self._load_labels()
        max_side = int(self.config.max_side)

        for path in pool:
            try:
                img = Image.open(path).convert("RGB")
                if max(img.size) > max_side * 2:
                    scale = (max_side * 2) / max(img.size)
                    img = img.resize(
                        (
                            max(1, int(round(img.size[0] * scale))),
                            max(1, int(round(img.size[1] * scale))),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                vec = self.encoder.encode(img)
            except Exception:
                continue
            entry = _BankEntry(vector=vec, path=str(path), kind="real", role="full")
            banks.real_full.append(entry)
            banks.neighbor_full.append(entry)

            if self.config.use_local:
                boxes = list(labels.get(path.name) or labels.get(path.stem) or [])
                class_boxes = [b for b in boxes if _label_matches(b.label, class_id)]
                if not class_boxes:
                    continue
                class_boxes.sort(
                    key=lambda b: max(0.0, float(b.bbox_xyxy[2]) - float(b.bbox_xyxy[0]))
                    * max(0.0, float(b.bbox_xyxy[3]) - float(b.bbox_xyxy[1])),
                    reverse=True,
                )
                try:
                    full = Image.open(path).convert("RGB")
                    crop = _crop_box(
                        full, class_boxes[0].bbox_xyxy, pad=0.08, max_side=max_side
                    )
                    cvec = self.encoder.encode(crop)
                    banks.real_crop.append(
                        _BankEntry(
                            vector=cvec, path=str(path), kind="real", role="crop"
                        )
                    )
                except Exception:
                    continue
        return banks

    def register_accepted(
        self,
        class_id: str,
        image: Image.Image,
        *,
        path: str | Path | None = None,
        crop: Image.Image | None = None,
    ) -> None:
        """Add an accepted synth to the novelty bank (full-frame)."""
        if not self.config.enabled:
            return
        with self._lock:
            banks = self._ensure_real_bank(str(class_id))
            try:
                vec = self.encoder.encode(image)
            except Exception:
                return
            entry = _BankEntry(
                vector=vec,
                path=str(path or f"synth:{class_id}:{len(banks.neighbor_full)}"),
                kind="synth",
                role="full",
            )
            banks.neighbor_full.append(entry)
            # Cap growth so long runs stay cheap.
            max_n = int(self.config.max_bank) * 2
            if len(banks.neighbor_full) > max_n:
                # Keep all reals, trim oldest synths.
                reals = [e for e in banks.neighbor_full if e.kind == "real"]
                synths = [e for e in banks.neighbor_full if e.kind == "synth"]
                banks.neighbor_full = reals + synths[-(max_n - len(reals)) :]

    def evaluate(
        self,
        image: Image.Image,
        class_id: str,
        *,
        crop: Image.Image | None = None,
        exclude_paths: set[str] | None = None,
    ) -> EmbeddingGateMetrics:
        cfg = self.config
        metrics = EmbeddingGateMetrics(enabled=cfg.enabled, model_id=cfg.model_id)
        if not cfg.enabled:
            return metrics

        with self._lock:
            banks = self._ensure_real_bank(str(class_id))
            exclude = {str(p) for p in (exclude_paths or set())}

            def _filter(entries: list[_BankEntry]) -> list[_BankEntry]:
                if not exclude:
                    return list(entries)
                return [
                    e
                    for e in entries
                    if Path(e.path).stem not in exclude and e.path not in exclude
                ]

            try:
                full_vec = self.encoder.encode(image)
            except Exception as exc:
                metrics.reason = f"encode-failed: {exc}"
                return metrics

            real_full = _filter(banks.real_full)
            neighbor_full = _filter(banks.neighbor_full)
            sim_g, path_g = nearest_neighbor(full_vec, real_full)
            sim_n, path_n = nearest_neighbor(full_vec, neighbor_full)
            metrics.real_sim_global = sim_g
            metrics.neighbor_sim = sim_n
            metrics.nearest_real = path_g
            metrics.nearest_neighbor = path_n

            reasons: list[str] = []
            if sim_g is None:
                pass
            elif float(sim_g) < float(cfg.min_real_sim_global):
                metrics.failed_fidelity = True
                reasons.append(
                    f"real_sim_global={sim_g:.3f} < min={cfg.min_real_sim_global:.3f}"
                )

            if cfg.use_local and crop is not None and banks.real_crop:
                try:
                    crop_vec = self.encoder.encode(crop)
                    sim_l, _ = nearest_neighbor(crop_vec, _filter(banks.real_crop))
                    metrics.real_sim_local = sim_l
                    if sim_l is not None and float(sim_l) < float(cfg.min_real_sim_local):
                        metrics.failed_fidelity = True
                        reasons.append(
                            f"real_sim_local={sim_l:.3f} < min={cfg.min_real_sim_local:.3f}"
                        )
                except Exception:
                    pass

            if sim_n is not None and float(sim_n) > float(cfg.max_neighbor_sim):
                metrics.failed_novelty = True
                reasons.append(
                    f"neighbor_sim={sim_n:.3f} > max={cfg.max_neighbor_sim:.3f}"
                )

            metrics.reason = "; ".join(reasons)
            return metrics


def embedding_gate_from_config(
    judge_cfg: Any,
    *,
    samples_dir: Path | str | None = None,
    stem_prefixes: list[str] | None = None,
    device: str | None = None,
) -> EmbeddingGate:
    raw = {}
    if judge_cfg is not None and hasattr(judge_cfg, "get"):
        raw = dict(judge_cfg.get("embedding_gate") or {})
    cfg = EmbeddingGateConfig(
        enabled=bool(raw.get("enabled", True)),
        model_id=str(raw.get("model_id", "openai/clip-vit-base-patch32")),
        min_real_sim_global=float(raw.get("min_real_sim_global", 0.58)),
        min_real_sim_local=float(raw.get("min_real_sim_local", 0.52)),
        max_neighbor_sim=float(raw.get("max_neighbor_sim", 0.94)),
        use_local=bool(raw.get("use_local", True)),
        fail_action=str(raw.get("fail_action", "retry")).lower(),
        max_bank=int(raw.get("max_bank", 256)),
        max_side=int(raw.get("max_side", 224)),
    )
    return EmbeddingGate(
        cfg,
        samples_dir=samples_dir,
        stem_prefixes=stem_prefixes,
        device=device,
    )
