"""VLM-as-judge for synthetic edge-case images.

Scores prompt faithfulness, physical plausibility, and (optionally) annotation
sanity. Returns accept / retry / reject using ``judge.threshold``.

Backends
--------
- ``qwen_vl`` — Qwen2.5-VL (3B on CPU profile, 7B on gpu_l4). Preferred.
- ``clip`` — CLIP similarity delta fallback when a local VLM is too heavy.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from edgecase_synthesis.conditioning import resolve_device


@dataclass
class JudgeResult:
    """Structured quality verdict for one synthetic image."""

    prompt_faithfulness: float
    physical_plausibility: float
    annotation_correctness: float
    edge_case_present: bool
    overall: float
    decision: str  # accept | retry | reject
    rationale: str
    raw_response: str = ""
    backend: str = ""
    model_id: str = ""
    anomaly_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_SCHEMA_HINT = """
Respond with ONLY a single JSON object (no markdown fences) using this schema:
{
  "prompt_faithfulness": <number 0-10>,
  "physical_plausibility": <number 0-10>,
  "annotation_correctness": <number 0-10>,
  "edge_case_present": <true|false>,
  "overall": <number 0-10>,
  "rationale": "<one short sentence>"
}
""".strip()


class VLMJudge:
    """Local vision-language judge (Qwen2.5-VL) with optional CLIP fallback."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        backend: str = "qwen_vl",
        device: str | None = None,
        torch_dtype: str = "float32",
        threshold: float = 8.5,
        reject_margin: float = 2.0,
        max_new_tokens: int = 320,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        schema_hint: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.backend = str(backend).lower()
        self.device = resolve_device(device)
        self.torch_dtype = _resolve_dtype(torch_dtype, self.device)
        self.threshold = float(threshold)
        self.reject_margin = float(reject_margin)
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.schema_hint = (schema_hint or _DEFAULT_SCHEMA_HINT).strip()
        self._model = None
        self._processor = None
        self._clip = None
        self._active_backend = self.backend

    def unload(self) -> None:
        """Drop weights so generation / annotation can reclaim VRAM."""
        self._model = None
        self._processor = None
        self._clip = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_qwen(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers with VLM support is required for backend=qwen_vl"
            ) from exc

        dtype_kw: dict[str, Any] = {}
        # Prefer dtype= for newer transformers; torch_dtype still widely accepted.
        dtype_kw["torch_dtype"] = self.torch_dtype

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map="auto" if self.device.type == "cuda" else None,
            **dtype_kw,
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        model.eval()
        self._processor = processor
        self._model = model
        self._active_backend = "qwen_vl"

    def _ensure_clip(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        clip_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(clip_id)
        model = CLIPModel.from_pretrained(clip_id)
        model.to(self.device).eval()
        self._clip = (processor, model)
        self._active_backend = "clip"
        self.model_id = clip_id

    @torch.inference_mode()
    def judge(
        self,
        image: Image.Image | Path | str,
        *,
        prompt: str,
        anomaly_id: str | None = None,
        anomaly_name: str | None = None,
        annotations_summary: str | None = None,
        source_hint: str | None = None,
    ) -> JudgeResult:
        """Score one synthetic RGB image (depth/seg maps are intentionally unused)."""
        pil = _to_pil(image)
        rare = anomaly_name or anomaly_id or "the rare edge-case condition"
        ann = annotations_summary or "No auto-annotations provided."
        src = source_hint or "a real photograph"

        if self.backend == "clip":
            self._ensure_clip()
            result = self._judge_clip(pil, prompt=prompt, rare=rare)
        else:
            try:
                self._ensure_qwen()
                result = self._judge_qwen(
                    pil,
                    prompt=prompt,
                    rare=rare,
                    annotations_summary=ann,
                    source_hint=src,
                )
            except Exception as exc:  # noqa: BLE001 — fall back so notebooks don't die
                # CPU machines without enough RAM / missing VLM deps still get a score.
                self._ensure_clip()
                result = self._judge_clip(pil, prompt=prompt, rare=rare)
                result.rationale = (
                    f"Qwen VLM unavailable ({type(exc).__name__}: {exc}); "
                    f"used CLIP fallback. {result.rationale}"
                )

        result.anomaly_id = anomaly_id
        result = self._reconcile(result)
        result.decision = self._decide(result)
        result.backend = self._active_backend
        result.model_id = self.model_id
        return result

    def _judge_qwen(
        self,
        image: Image.Image,
        *,
        prompt: str,
        rare: str,
        annotations_summary: str,
        source_hint: str,
    ) -> JudgeResult:
        assert self._model is not None and self._processor is not None
        user_text = (
            f"You are a strict data-quality judge for synthetic training images.\n"
            f"The image was edited from {source_hint}.\n"
            f"Target rare condition: {rare}\n"
            f"Generation prompt:\n{prompt}\n\n"
            f"Auto-annotation summary:\n{annotations_summary}\n\n"
            f"Consistency rules:\n"
            f"- If the target rare condition is visible, edge_case_present MUST be true.\n"
            f"- If prompt_faithfulness ≥ 7, the rare condition is present → "
            f"edge_case_present must be true.\n"
            f"- overall ≥ 8 means you would keep this sample for training.\n\n"
            f"{self.schema_hint}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # Trim prompt tokens when present.
        if "input_ids" in inputs:
            trimmed = [
                out[len(inp) :]
                for inp, out in zip(inputs["input_ids"], generated_ids, strict=True)
            ]
            text = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        parsed = _parse_judge_json(text)
        return _result_from_parsed(parsed, raw_response=text)

    def _judge_clip(
        self,
        image: Image.Image,
        *,
        prompt: str,
        rare: str,
    ) -> JudgeResult:
        assert self._clip is not None
        processor, model = self._clip
        positive = f"a photo of {rare} on a street or road"
        negative = "a normal empty road with no obstruction"
        prompt_text = prompt.strip() or positive

        inputs = processor(
            text=[positive, negative, prompt_text],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            dim=-1, keepdim=True
        )
        sims = (image_embeds @ text_embeds.T).squeeze(0)
        pos, neg, prompt_sim = (float(x) for x in sims.tolist())
        delta = pos - neg
        # Map CLIP delta roughly onto 0–10 (heuristic, not calibrated).
        faithfulness = _clip_to_10(prompt_sim, lo=0.15, hi=0.35)
        presence_score = _clip_to_10(delta, lo=-0.02, hi=0.12)
        plausibility = 0.55 * faithfulness + 0.45 * presence_score
        edge = delta > 0.02 and pos > neg
        overall = 0.45 * faithfulness + 0.35 * plausibility + 0.20 * presence_score
        if not edge:
            overall = min(overall, 4.5)
        return JudgeResult(
            prompt_faithfulness=round(faithfulness, 2),
            physical_plausibility=round(plausibility, 2),
            annotation_correctness=5.0,
            edge_case_present=bool(edge),
            overall=round(overall, 2),
            decision="retry",  # filled by _decide
            rationale=(
                f"CLIP Δ(rare−normal)={delta:.3f}; "
                f"sim(prompt)={prompt_sim:.3f}. Heuristic fallback, not a full VLM."
            ),
            raw_response=json.dumps(
                {"pos": pos, "neg": neg, "prompt_sim": prompt_sim, "delta": delta}
            ),
        )

    def _reconcile(self, result: JudgeResult) -> JudgeResult:
        """Fix common VLM inconsistencies before the accept/retry/reject gate.

        Qwen sometimes sets edge_case_present=false while scoring faithfulness
        high and writing that the object is visible — that would hard-reject.
        """
        looks_present = (
            result.prompt_faithfulness >= 7.0
            or result.overall >= self.threshold
        )
        if not result.edge_case_present and looks_present:
            result.edge_case_present = True
            note = " [reconciled: edge_case_present set true from high scores]"
            if note.strip() not in result.rationale:
                result.rationale = (result.rationale or "").rstrip() + note
        if result.edge_case_present and result.overall < 3.0:
            # Rare: flagged present but abysmal overall — leave as-is for reject/retry.
            pass
        return result

    def _decide(self, result: JudgeResult) -> str:
        """Gate on overall vs threshold (accept / retry / reject).

        With threshold=8.0 and reject_margin=2.0:
        - overall ≥ 8 and edge present → accept
        - overall < 6 or no edge → reject
        - otherwise → retry
        """
        if (not result.edge_case_present) or result.overall < (
            self.threshold - self.reject_margin
        ):
            return "reject"
        if result.overall >= self.threshold and result.edge_case_present:
            return "accept"
        return "retry"

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | Any, device: str | None = None):
        judge = cfg.get("judge", cfg)
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            model_id=str(judge.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
            backend=str(judge.get("backend", "qwen_vl")),
            device=device,
            torch_dtype=str(judge.get("torch_dtype", "float32")),
            threshold=float(judge.get("threshold", 8.5)),
            reject_margin=float(judge.get("reject_margin", 2.0)),
            max_new_tokens=int(judge.get("max_new_tokens", 320)),
            min_pixels=int(judge.get("min_pixels", 256 * 28 * 28)),
            max_pixels=int(judge.get("max_pixels", 512 * 28 * 28)),
            schema_hint=judge.get("schema_hint"),
        )


def summarize_annotations(annotation: Any | None) -> str:
    """Compact text summary for the judge prompt (not masks / depth)."""
    if annotation is None:
        return "No auto-annotations provided."
    detections = getattr(annotation, "detections", None) or []
    if not detections:
        return "Detector returned zero instances."
    parts = []
    for det in detections[:12]:
        label = getattr(det, "label", "?")
        conf = float(getattr(det, "confidence", 0.0))
        box = getattr(det, "bbox_xyxy", None)
        if box is not None:
            parts.append(f"{label} conf={conf:.2f} bbox={tuple(int(v) for v in box)}")
        else:
            parts.append(f"{label} conf={conf:.2f}")
    return "; ".join(parts)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    key = str(name).lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16 if device.type == "cuda" else torch.float32
    if key in {"bfloat16", "bf16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


def _to_pil(image: Image.Image | Path | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _clip_to_10(value: float, *, lo: float, hi: float) -> float:
    if hi <= lo:
        return 5.0
    t = (float(value) - lo) / (hi - lo)
    return float(max(0.0, min(10.0, 10.0 * t)))


def _parse_judge_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            raw = match.group(0)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {
        "prompt_faithfulness": 3.0,
        "physical_plausibility": 3.0,
        "annotation_correctness": 3.0,
        "edge_case_present": False,
        "overall": 3.0,
        "rationale": f"Failed to parse VLM JSON; raw starts: {text[:160]!r}",
    }


def _result_from_parsed(parsed: dict[str, Any], *, raw_response: str) -> JudgeResult:
    def _num(key: str, default: float = 0.0) -> float:
        try:
            return float(parsed.get(key, default))
        except (TypeError, ValueError):
            return default

    edge = parsed.get("edge_case_present", False)
    if isinstance(edge, str):
        edge = edge.strip().lower() in {"1", "true", "yes", "y"}
    overall = _num("overall", 0.0)
    if overall <= 0:
        scores = [
            _num("prompt_faithfulness"),
            _num("physical_plausibility"),
            _num("annotation_correctness"),
        ]
        overall = sum(scores) / max(len(scores), 1)
    return JudgeResult(
        prompt_faithfulness=_num("prompt_faithfulness"),
        physical_plausibility=_num("physical_plausibility"),
        annotation_correctness=_num("annotation_correctness", 5.0),
        edge_case_present=bool(edge),
        overall=round(overall, 2),
        decision="retry",
        rationale=str(parsed.get("rationale", "") or ""),
        raw_response=raw_response,
    )
