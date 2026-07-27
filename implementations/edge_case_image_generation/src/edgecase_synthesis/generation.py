"""Config-driven image editing: depth ControlNet and optional inpaint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from edgecase_synthesis.conditioning import (
    DepthResult,
    SegmentationResult,
    build_anomaly_edit_mask,
    resolve_device,
)


@dataclass
class GenerationResult:
    image: Image.Image
    prompt: str
    negative_prompt: str
    seed: int
    edit_mask: np.ndarray | None = None
    anomaly_id: str | None = None
    method: str | None = None


class AnomalyEditor:
    """Edit a real photo with ControlNet (depth) and/or inpaint.

    Method resolution: anomaly YAML ``method``, or ``auto`` →
    ``generation.default_anomaly_method`` from the hardware profile.
    """

    def __init__(
        self,
        *,
        family: str = "sd15",
        base_model_id: str,
        inpaint_model_id: str | None = None,
        depth_controlnet_id: str,
        vae_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.family = str(family).lower()
        self.base_model_id = base_model_id
        self.inpaint_model_id = inpaint_model_id
        self.depth_controlnet_id = depth_controlnet_id
        self.vae_id = vae_id
        self.device = resolve_device(device)
        self._controlnet_pipe = None
        self._inpaint_pipe = None

    # Back-compat name used in earlier notebook cells.
    @property
    def pipe(self):
        return self.controlnet_pipe

    @property
    def controlnet_pipe(self):
        if self._controlnet_pipe is None:
            self._controlnet_pipe = self._build_controlnet()
        return self._controlnet_pipe

    @property
    def inpaint_pipe(self):
        if self._inpaint_pipe is None:
            self._inpaint_pipe = self._build_inpaint()
        return self._inpaint_pipe

    def unload(self) -> None:
        self._controlnet_pipe = None
        self._inpaint_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def _place(self, pipe: Any) -> Any:
        pipe.set_progress_bar_config(disable=False)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if self.device.type == "cuda":
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
        else:
            pipe.to(self.device)
        return pipe

    def _build_controlnet(self):
        from diffusers import ControlNetModel

        dtype = self._dtype()
        depth_cn = ControlNetModel.from_pretrained(self.depth_controlnet_id, torch_dtype=dtype)
        if self.family == "sdxl":
            from diffusers import AutoencoderKL, StableDiffusionXLControlNetImg2ImgPipeline

            kwargs: dict[str, Any] = {
                "controlnet": depth_cn,
                "torch_dtype": dtype,
            }
            if self.vae_id:
                kwargs["vae"] = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.base_model_id, **kwargs
            )
            return self._place(pipe)

        from diffusers import StableDiffusionControlNetImg2ImgPipeline

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            controlnet=depth_cn,
            torch_dtype=dtype,
            safety_checker=None,
        )
        return self._place(pipe)

    def _build_inpaint(self):
        if not self.inpaint_model_id:
            raise ValueError("generation.inpaint_model_id required for method=inpaint")
        from diffusers import AutoPipelineForInpainting

        dtype = self._dtype()
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        return self._place(pipe)

    def generate_anomaly(
        self,
        image: Image.Image,
        depth: DepthResult,
        segmentation: SegmentationResult | None,
        generation_cfg: Any,
        anomaly_cfg: Any,
    ) -> GenerationResult:
        from edgecase_synthesis.config import merge_generation_anomaly

        merged = merge_generation_anomaly(generation_cfg, anomaly_cfg)
        anom = merged.get("anomaly", anomaly_cfg)
        method = _resolve_method(anom, merged)
        max_side = int(merged.get("max_side", 512))
        seed = int(merged.get("seed", 42))
        prompt = str(merged.get("prompt", ""))
        negative = str(merged.get("negative_prompt", ""))
        anomaly_id = str(anom.get("id", ""))
        edit_mask_cfg = dict(anom.get("edit_mask", {"mode": "ellipse"}))
        family = str(merged.get("family", self.family)).lower()

        original = _fit_for_diffusion(image.convert("RGB"), max_side=max_side, family=family)
        width, height = original.size
        edit_mask, edit_weight = build_anomaly_edit_mask(
            segmentation,
            edit_mask_cfg,
            width=width,
            height=height,
            depth=depth,
        )

        if method == "inpaint":
            return self._inpaint(
                original,
                prompt=prompt,
                negative_prompt=negative,
                steps=int(merged.get("num_inference_steps", 28)),
                guidance=float(merged.get("guidance_scale", 7.5)),
                strength=float(merged.get("strength", 0.90)),
                seed=seed,
                edit_mask=edit_mask,
                edit_weight=edit_weight,
                edit_mask_cfg=edit_mask_cfg,
                anomaly_id=anomaly_id,
                padding_mask_crop=(
                    int(merged["padding_mask_crop"])
                    if merged.get("padding_mask_crop") not in (None, "", False)
                    else None
                ),
            )

        scale_cfg = merged.get("controlnet_scale", 0.55)
        if isinstance(scale_cfg, (int, float)):
            cn_scale = float(scale_cfg)
        else:
            cn_scale = float(scale_cfg.get("depth", 0.75))

        cn_strength = float(
            merged.get("controlnet_strength", merged.get("strength", 0.45))
        )
        return self._controlnet(
            original,
            depth,
            prompt=prompt,
            negative_prompt=negative,
            steps=int(merged.get("num_inference_steps", 16)),
            guidance=float(merged.get("guidance_scale", 7.0)),
            strength=float(np.clip(cn_strength, 0.25, 0.60)),
            seed=seed,
            controlnet_scale=cn_scale,
            edit_mask=edit_mask,
            edit_weight=edit_weight,
            edit_mask_cfg=edit_mask_cfg,
            anomaly_id=anomaly_id,
        )

    @torch.inference_mode()
    def _controlnet(
        self,
        original: Image.Image,
        depth: DepthResult,
        *,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
        controlnet_scale: float,
        edit_mask: np.ndarray,
        edit_weight: np.ndarray,
        edit_mask_cfg: dict[str, Any],
        anomaly_id: str | None,
    ) -> GenerationResult:
        width, height = original.size
        depth_image = _depth_to_control_image(depth, width, height)
        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        generated = self.controlnet_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=original,
            control_image=depth_image,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=generator,
        ).images[0]
        generated = _ensure_same_size(generated, original)
        blur = float(edit_mask_cfg.get("composite_blur", 1.0))
        output = _composite(original, generated, edit_weight, blur_sigma=blur)
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            edit_mask=edit_mask,
            anomaly_id=anomaly_id,
            method="controlnet",
        )

    @torch.inference_mode()
    def _inpaint(
        self,
        original: Image.Image,
        *,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
        edit_mask: np.ndarray,
        edit_weight: np.ndarray,
        edit_mask_cfg: dict[str, Any],
        anomaly_id: str | None,
        padding_mask_crop: int | None = None,
    ) -> GenerationResult:
        width, height = original.size
        mask_pil = _mask_to_pil(edit_mask)
        if mask_pil.size != (width, height):
            mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "image": original,
            "mask_image": mask_pil,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": min(float(strength), 0.99),
            "generator": generator,
        }
        if padding_mask_crop is not None and padding_mask_crop > 0:
            kwargs["padding_mask_crop"] = int(padding_mask_crop)
        try:
            generated = self.inpaint_pipe(**kwargs).images[0]
        except TypeError:
            kwargs.pop("padding_mask_crop", None)
            generated = self.inpaint_pipe(**kwargs).images[0]
        generated = _ensure_same_size(generated, original)
        blur = float(edit_mask_cfg.get("composite_blur", 2.0))
        output = _composite(original, generated, edit_weight, blur_sigma=blur)
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            edit_mask=edit_mask,
            anomaly_id=anomaly_id,
            method="inpaint",
        )

    @classmethod
    def from_config(cls, cfg: Any, device: str | None = None):
        generation = cfg.get("generation", cfg)
        controlnet = generation.get("controlnet", {})
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        return cls(
            family=str(generation.get("family", "sd15")),
            base_model_id=str(generation["base_model_id"]),
            inpaint_model_id=generation.get("inpaint_model_id"),
            depth_controlnet_id=str(controlnet["depth"]),
            vae_id=generation.get("vae_id"),
            device=device,
        )


# Notebook back-compat
ControlNetEditor = AnomalyEditor


def resolve_anomaly_method(anom: Any, merged: Any) -> str:
    """Public helper for notebooks: anomaly method or hardware default."""
    return _resolve_method(anom, merged)


def _resolve_method(anom: Any, merged: Any) -> str:
    raw = anom.get("method") if hasattr(anom, "get") else None
    if raw is None or str(raw).lower() in {"", "auto", "default"}:
        return str(merged.get("default_anomaly_method", "inpaint")).lower()
    return str(raw).lower()


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    u8 = (np.asarray(mask).astype(np.float32) > 0.5).astype(np.uint8) * 255
    if u8.any():
        k = max(3, int(round(0.01 * max(u8.shape))))
        if k % 2 == 0:
            k += 1
        u8 = cv2.dilate(u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return Image.fromarray(u8, mode="L")


def _depth_to_control_image(depth: DepthResult, width: int, height: int) -> Image.Image:
    """Grayscale depth for ControlNet (near=bright). Avoid feeding the viz colormap."""
    d = np.asarray(depth.depth_map, dtype=np.float32)
    if d.shape != (height, width):
        d = cv2.resize(d, (width, height), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(d.min()), float(d.max())
    if hi > lo:
        d = (d - lo) / (hi - lo)
    else:
        d = np.zeros_like(d)
    u8 = (np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _composite(
    original: Image.Image,
    generated: Image.Image,
    weight: np.ndarray,
    *,
    blur_sigma: float,
) -> Image.Image:
    generated = _ensure_same_size(generated, original)
    base = np.array(original.convert("RGB"), dtype=np.float32)
    gen = np.array(generated.convert("RGB"), dtype=np.float32)
    h, w = base.shape[:2]
    weight_arr = np.asarray(weight, dtype=np.float32)
    if weight_arr.shape[:2] != (h, w):
        weight_arr = cv2.resize(weight_arr, (w, h), interpolation=cv2.INTER_LINEAR)
    soft = np.clip(cv2.GaussianBlur(weight_arr, (0, 0), max(blur_sigma, 0.5)), 0, 1)[..., None]
    out = base * (1.0 - soft) + gen * soft
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _ensure_same_size(image: Image.Image, reference: Image.Image) -> Image.Image:
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.Resampling.LANCZOS)


def _fit_for_diffusion(image: Image.Image, *, max_side: int, family: str) -> Image.Image:
    fitted = _fit_max_side(image, max_side)
    if family != "sdxl":
        return fitted
    min_long = min(int(max_side), 1024)
    long_side = max(fitted.size)
    if long_side >= min_long:
        return fitted
    scale = min_long / long_side
    new_w = _round_to_multiple(max(8, int(round(fitted.size[0] * scale))), 8)
    new_h = _round_to_multiple(max(8, int(round(fitted.size[1] * scale))), 8)
    return fitted.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _fit_max_side(image: Image.Image, max_side: int | None) -> Image.Image:
    w, h = image.size
    if max_side is None or max(w, h) <= max_side:
        return image.resize(
            (_round_to_multiple(w, 8), _round_to_multiple(h, 8)),
            Image.Resampling.LANCZOS,
        )
    scale = max_side / max(w, h)
    return image.resize(
        (
            _round_to_multiple(max(8, int(round(w * scale))), 8),
            _round_to_multiple(max(8, int(round(h * scale))), 8),
        ),
        Image.Resampling.LANCZOS,
    )


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def _resize_rgb(image: np.ndarray, width: int, height: int) -> Image.Image:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return pil
