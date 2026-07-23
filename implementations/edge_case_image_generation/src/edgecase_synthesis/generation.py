"""Controlled image editing for edge-case anomalies.

Methods (per anomaly YAML, or ``auto`` → ``generation.default_anomaly_method``):

- ``paste`` — render a cone/branch/animal layer and composite (CPU / MPS default).
- ``inpaint`` — SDXL inpaint after optional silhouette priming (GPU L4 default).
- ``controlnet`` — depth ControlNet img2img (SD 1.5 or SDXL per ``generation.family``).
  Best for appearance/weather edits (e.g. snow) that must respect geometry.

Depth ControlNet *fights* inventing new objects on the track plane — prefer paste
or inpaint for inserts, not controlnet.
"""

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
    """Output of a single anomaly edit."""

    image: Image.Image
    prompt: str
    negative_prompt: str
    seed: int
    edit_mask: np.ndarray | None = None  # bool (H, W) at output resolution
    anomaly_id: str | None = None
    method: str | None = None


class ControlNetEditor:
    """Anomaly editor: paste, SDXL inpaint, and/or depth ControlNet (SD 1.5 / SDXL)."""

    def __init__(
        self,
        *,
        family: str = "sd15",
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        inpaint_model_id: str | None = None,
        depth_controlnet_id: str = "lllyasviel/sd-controlnet-depth",
        seg_controlnet_id: str | None = "lllyasviel/sd-controlnet-seg",
        use_seg: bool = False,
        device: str | None = None,
    ) -> None:
        self.family = str(family).lower()
        self.base_model_id = base_model_id
        self.inpaint_model_id = inpaint_model_id
        self.depth_controlnet_id = depth_controlnet_id
        self.seg_controlnet_id = seg_controlnet_id
        self.use_seg = bool(use_seg) and bool(seg_controlnet_id)
        self.device = resolve_device(device)
        self._controlnet_pipe = None
        self._inpaint_pipe = None

    @property
    def pipe(self):
        """Lazy ControlNet img2img pipeline (weather / appearance)."""
        if self._controlnet_pipe is None:
            self._controlnet_pipe = self._build_controlnet_pipeline()
        return self._controlnet_pipe

    @property
    def inpaint_pipe(self):
        """Lazy inpaint pipeline (object inserts on GPU)."""
        if self._inpaint_pipe is None:
            self._inpaint_pipe = self._build_inpaint_pipeline()
        return self._inpaint_pipe

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def _place_pipe(self, pipe: Any) -> Any:
        pipe.set_progress_bar_config(disable=False)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if self.device.type == "cuda":
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            # Offload keeps peak VRAM manageable on L4 when stacking CN + VAE.
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
        else:
            pipe.to(self.device)
        return pipe

    def _build_controlnet_pipeline(self):
        from diffusers import ControlNetModel

        dtype = self._dtype()
        depth_cn = ControlNetModel.from_pretrained(
            self.depth_controlnet_id,
            torch_dtype=dtype,
        )

        if self.family == "sdxl":
            from diffusers import (
                AutoencoderKL,
                StableDiffusionXLControlNetImg2ImgPipeline,
            )

            controlnet: Any = depth_cn
            if self.use_seg and self.seg_controlnet_id:
                from diffusers import MultiControlNetModel

                seg_cn = ControlNetModel.from_pretrained(
                    self.seg_controlnet_id,
                    torch_dtype=dtype,
                )
                controlnet = MultiControlNetModel([depth_cn, seg_cn])

            # fp16-fixed VAE avoids NaNs with SDXL on CUDA.
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=dtype,
            )
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.base_model_id,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=dtype,
            )
            return self._place_pipe(pipe)

        from diffusers import (
            MultiControlNetModel,
            StableDiffusionControlNetImg2ImgPipeline,
        )

        if self.use_seg and self.seg_controlnet_id:
            seg_cn = ControlNetModel.from_pretrained(
                self.seg_controlnet_id,
                torch_dtype=dtype,
            )
            controlnet = MultiControlNetModel([depth_cn, seg_cn])
        else:
            controlnet = depth_cn

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        return self._place_pipe(pipe)

    def _build_inpaint_pipeline(self):
        if not self.inpaint_model_id:
            raise ValueError(
                "generation.inpaint_model_id is required for method=inpaint "
                "(use hardware=gpu_l4 or set an SDXL inpaint checkpoint)."
            )
        from diffusers import AutoPipelineForInpainting

        dtype = self._dtype()
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        return self._place_pipe(pipe)

    def generate_anomaly(
        self,
        image: Image.Image,
        depth: DepthResult,
        segmentation: SegmentationResult | None,
        generation_cfg: Any,
        anomaly_cfg: Any,
    ) -> GenerationResult:
        """Run one anomaly using merged generation + anomaly YAML settings."""
        from edgecase_synthesis.config import merge_generation_anomaly

        merged = merge_generation_anomaly(generation_cfg, anomaly_cfg)
        anom = merged.get("anomaly", anomaly_cfg)
        method = _resolve_method(anom, merged)
        max_side = int(merged.get("max_side", 512))
        seed = int(merged.get("seed", 42))
        prompt = str(merged.get("prompt", ""))
        negative = str(merged.get("negative_prompt", ""))
        anomaly_id = str(anom.get("id", ""))
        edit_mask_cfg = dict(anom.get("edit_mask", {"mode": "blob"}))
        prime_cfg = dict(anom.get("prime", {})) if anom.get("prime") is not None else {}

        original = _fit_max_side(image.convert("RGB"), max_side)
        width, height = original.size
        edit_mask, edit_weight = build_anomaly_edit_mask(
            segmentation,
            edit_mask_cfg,
            width=width,
            height=height,
        )

        if method in {"paste", "composite"}:
            output, obj_mask = _paste_object(original, edit_mask, prime_cfg, seed=seed)
            return GenerationResult(
                image=output,
                prompt=prompt,
                negative_prompt=negative,
                seed=seed,
                edit_mask=obj_mask,
                anomaly_id=anomaly_id,
                method="paste",
            )

        if method == "inpaint":
            return self._generate_inpaint(
                original,
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=int(merged.get("num_inference_steps", 28)),
                guidance_scale=float(merged.get("guidance_scale", 7.0)),
                strength=float(merged.get("strength", 0.88)),
                seed=seed,
                edit_mask=edit_mask,
                edit_weight=edit_weight,
                edit_mask_cfg=edit_mask_cfg,
                prime_cfg=prime_cfg or None,
                anomaly_id=anomaly_id,
            )

        # controlnet / img2img path (weather & appearance edits)
        return self._generate_controlnet(
            original,
            depth,
            segmentation,
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(merged.get("num_inference_steps", 16)),
            guidance_scale=float(merged.get("guidance_scale", 7.5)),
            strength=float(merged.get("strength", 0.55)),
            seed=seed,
            controlnet_scale=merged.get("controlnet_scale", {"depth": 0.5, "seg": 0.35}),
            edit_mask=edit_mask,
            edit_weight=edit_weight,
            edit_mask_cfg=edit_mask_cfg,
            prime_cfg=prime_cfg or None,
            snow_prime=merged.get("snow_prime"),
            winter_grade=merged.get("winter_grade"),
            anomaly_id=anomaly_id,
        )

    @torch.inference_mode()
    def _generate_inpaint(
        self,
        original: Image.Image,
        *,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        edit_mask: np.ndarray,
        edit_weight: np.ndarray,
        edit_mask_cfg: dict[str, Any],
        prime_cfg: dict[str, Any] | None,
        anomaly_id: str | None,
    ) -> GenerationResult:
        working = original
        obj_mask = edit_mask
        if prime_cfg:
            # Silhouette primes the inpaint region so the model has a shape prior.
            working, obj_mask = _paste_object(working, edit_mask, prime_cfg, seed=seed)

        width, height = working.size
        mask_pil = _mask_to_pil(obj_mask if obj_mask is not None else edit_mask)
        if mask_pil.size != (width, height):
            mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        # SDXL defaults to 1024×1024 if width/height are omitted — must match the photo.
        generated = self.inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=working,
            mask_image=mask_pil,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]
        generated = _ensure_same_size(generated, working)

        blur = float(edit_mask_cfg.get("composite_blur", 1.5))
        # Soft-lock unmasked pixels to the (primed) photo.
        weight = edit_weight if edit_weight is not None else obj_mask.astype(np.float32)
        if obj_mask is not None:
            weight = np.maximum(weight.astype(np.float32), obj_mask.astype(np.float32))
        output = _composite_with_weight(working, generated, weight, blur_sigma=blur)
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            edit_mask=obj_mask if obj_mask is not None else edit_mask,
            anomaly_id=anomaly_id,
            method="inpaint",
        )

    @torch.inference_mode()
    def _generate_controlnet(
        self,
        original: Image.Image,
        depth: DepthResult,
        segmentation: SegmentationResult | None,
        *,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        controlnet_scale: Any,
        edit_mask: np.ndarray,
        edit_weight: np.ndarray,
        edit_mask_cfg: dict[str, Any],
        prime_cfg: dict[str, Any] | None,
        snow_prime: float | None,
        winter_grade: float | None,
        anomaly_id: str | None,
    ) -> GenerationResult:
        width, height = original.size
        depth_image = _resize_rgb(depth.colormap, width, height)

        working = original
        if winter_grade is not None and float(winter_grade) > 0:
            winter_weight = _prepare_winter_weight(segmentation, width, height)
            working = _apply_winter_grade(working, winter_weight, amount=float(winter_grade))
        if snow_prime is not None and float(snow_prime) > 0:
            working = _prime_snow(working, edit_weight, mix=float(snow_prime))
        if prime_cfg:
            working, _ = _paste_object(working, edit_mask, prime_cfg, seed=seed)

        scales = _resolve_controlnet_scales(controlnet_scale, use_seg=self.use_seg)
        if self.use_seg:
            seg_image = (
                _resize_rgb(segmentation.colored_map, width, height)
                if segmentation is not None
                else Image.new("RGB", (width, height), (0, 0, 0))
            )
            control_image: Any = [depth_image, seg_image]
        else:
            control_image = depth_image

        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        generated = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=working,
            control_image=control_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=scales,
            generator=generator,
        ).images[0]
        generated = _ensure_same_size(generated, working)

        blur = float(edit_mask_cfg.get("composite_blur", 1.5))
        output = _composite_with_weight(working, generated, edit_weight, blur_sigma=blur)
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            edit_mask=edit_mask,
            anomaly_id=anomaly_id,
            method="controlnet",
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | Any, device: str | None = None):
        generation = cfg.get("generation", cfg)
        controlnet = generation.get("controlnet", {})
        return cls(
            family=str(generation.get("family", "sd15")),
            base_model_id=generation.get("base_model_id", "runwayml/stable-diffusion-v1-5"),
            inpaint_model_id=generation.get("inpaint_model_id"),
            depth_controlnet_id=controlnet.get("depth", "lllyasviel/sd-controlnet-depth"),
            seg_controlnet_id=controlnet.get("seg"),
            use_seg=bool(controlnet.get("use_seg", False)),
            device=_device_from_cfg(cfg, device),
        )


def _device_from_cfg(cfg: Any, device: str | None) -> str | None:
    if device is not None:
        return device
    hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
    if hardware is None:
        return None
    return hardware.get("device")


def _resolve_method(anom: Any, merged: Any) -> str:
    raw = anom.get("method") if hasattr(anom, "get") else None
    if raw is None or str(raw).lower() in {"", "auto", "default"}:
        return str(merged.get("default_anomaly_method", "paste")).lower()
    return str(raw).lower()


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    """White = inpaint region (diffusers convention)."""
    u8 = (np.asarray(mask).astype(np.float32) > 0.5).astype(np.uint8) * 255
    # Slight dilate so the silhouette edge gets cleaned by the inpaint UNet.
    if u8.any():
        k = max(3, int(round(0.01 * max(u8.shape))))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        u8 = cv2.dilate(u8, kernel, iterations=1)
    return Image.fromarray(u8, mode="L")


def _paste_object(
    image: Image.Image,
    region_mask: np.ndarray,
    prime_cfg: dict[str, Any],
    *,
    seed: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """Composite a rendered object into ``region_mask`` with a soft ground shadow."""
    shape = str(prime_cfg.get("shape", "cone")).lower()
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = region_mask.shape
    if not region_mask.any():
        return image, region_mask

    ys, xs = np.where(region_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Use the region bbox; grow slightly for shadow room.
    pad = max(4, int(0.04 * (y1 - y0 + 1)))
    y0, y1 = max(0, y0 - pad), min(h - 1, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w - 1, x1 + pad)
    bh, bw = y1 - y0 + 1, x1 - x0 + 1

    layer = np.zeros((bh, bw, 4), dtype=np.float32)  # RGBA
    if shape == "cone":
        _draw_cone(layer, prime_cfg)
    elif shape == "branch":
        _draw_branch(layer, prime_cfg)
    else:
        _draw_animal(layer, prime_cfg, kind=shape, seed=seed)

    # Soft shadow under the object (ellipse near bottom).
    shadow = np.zeros((bh, bw), dtype=np.float32)
    scx, scy = 0.50 * bw, 0.88 * bh
    srx, sry = 0.28 * bw, 0.10 * bh
    yy, xx = np.mgrid[0:bh, 0:bw]
    shadow = np.clip(1.0 - (((xx - scx) / max(srx, 1)) ** 2 + ((yy - scy) / max(sry, 1)) ** 2), 0, 1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(bw * 0.03, 1.0)) * 0.45

    out = rgb.copy()
    roi = out[y0 : y1 + 1, x0 : x1 + 1]
    # Darken for shadow first.
    roi *= 1.0 - shadow[..., None] * 0.55
    alpha = np.clip(layer[..., 3:4] / 255.0, 0.0, 1.0)
    roi[:] = roi * (1.0 - alpha) + layer[..., :3] * alpha

    obj_mask = np.zeros((h, w), dtype=bool)
    obj_mask[y0 : y1 + 1, x0 : x1 + 1] = layer[..., 3] > 20
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), obj_mask


def _draw_cone(layer: np.ndarray, prime_cfg: dict[str, Any]) -> None:
    """Perspective-ish traffic cone (wide base, pointed top) with white stripe."""
    bh, bw = layer.shape[:2]
    base = np.array(prime_cfg.get("color", [255, 110, 30]), dtype=np.float32)
    for y in range(bh):
        t = y / max(bh - 1, 1)  # 0 top → 1 bottom
        # Skip upper empty sky of the bbox a bit; cone occupies mid-lower.
        if t < 0.12:
            continue
        u = (t - 0.12) / 0.88
        half = (0.06 + 0.38 * u) * bw
        cx = 0.50 * bw
        x_left, x_right = int(cx - half), int(cx + half)
        # Vertical shading (darker on the right face).
        for x in range(max(0, x_left), min(bw, x_right + 1)):
            face = (x - x_left) / max(x_right - x_left, 1)
            shade = 0.75 + 0.35 * (1.0 - abs(face - 0.35))
            color = np.clip(base * shade, 0, 255)
            layer[y, x, :3] = color
            layer[y, x, 3] = 255
    # White reflective band.
    y_a, y_b = int(0.48 * bh), int(0.58 * bh)
    band = layer[y_a:y_b, :, 3] > 0
    layer[y_a:y_b, :, :3][band] = np.array([245, 245, 240], dtype=np.float32)
    # Soften edges.
    alpha = layer[..., 3]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    layer[..., 3] = alpha


def _draw_branch(layer: np.ndarray, prime_cfg: dict[str, Any]) -> None:
    """Thick tapered branch across the strip."""
    bh, bw = layer.shape[:2]
    base = np.array(prime_cfg.get("color", [95, 58, 32]), dtype=np.float32)
    cy = 0.55 * bh
    for x in range(bw):
        t = x / max(bw - 1, 1)
        # Slight curve + taper toward ends.
        y_off = 0.08 * bh * np.sin(t * np.pi)
        thick = (0.18 + 0.22 * np.sin(t * np.pi)) * bh
        y_mid = cy + y_off
        y0, y1 = int(y_mid - thick), int(y_mid + thick)
        for y in range(max(0, y0), min(bh, y1 + 1)):
            v = abs((y - y_mid) / max(thick, 1))
            shade = 0.65 + 0.45 * (1.0 - v)
            # Bark noise
            noise = 0.92 + 0.08 * np.sin(x * 0.35 + y * 0.2)
            layer[y, x, :3] = np.clip(base * shade * noise, 0, 255)
            layer[y, x, 3] = 255 * (1.0 - v * 0.15)
    layer[..., 3] = cv2.GaussianBlur(layer[..., 3], (0, 0), 1.0)


def _draw_animal(
    layer: np.ndarray,
    prime_cfg: dict[str, Any],
    *,
    kind: str,
    seed: int,
) -> None:
    """Simple side-view animal silhouette (body + head + legs)."""
    bh, bw = layer.shape[:2]
    base = np.array(prime_cfg.get("color", [120, 80, 45]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    # Body ellipse
    body = np.zeros((bh, bw), dtype=np.float32)
    yy, xx = np.mgrid[0:bh, 0:bw]
    bcx, bcy = 0.48 * bw, 0.58 * bh
    brx, bry = 0.32 * bw, 0.22 * bh
    body += ((((xx - bcx) / max(brx, 1)) ** 2 + ((yy - bcy) / max(bry, 1)) ** 2) <= 1.0).astype(
        np.float32
    )
    # Head
    if "deer" in kind:
        hcx, hcy = 0.78 * bw, 0.42 * bh
        hrx, hry = 0.12 * bw, 0.14 * bh
        # Simple antler stubs
        for dx, dy in [(0.82, 0.22), (0.88, 0.18), (0.76, 0.20)]:
            ax, ay = dx * bw, dy * bh
            body[((xx - ax) ** 2 + (yy - ay) ** 2) < (0.025 * bw) ** 2] = 1.0
    else:  # dog / generic
        hcx, hcy = 0.78 * bw, 0.48 * bh
        hrx, hry = 0.11 * bw, 0.11 * bh
    body += ((((xx - hcx) / max(hrx, 1)) ** 2 + ((yy - hcy) / max(hry, 1)) ** 2) <= 1.0).astype(
        np.float32
    )
    # Legs
    for lx in (0.30, 0.42, 0.58, 0.68):
        leg_x = lx * bw
        leg = (np.abs(xx - leg_x) < 0.035 * bw) & (yy > bcy) & (yy < 0.92 * bh)
        body[leg] = 1.0
    body = np.clip(body, 0, 1)
    body = cv2.GaussianBlur(body, (0, 0), 1.0)
    shade = 0.85 + 0.15 * rng.random((bh, bw))
    for c in range(3):
        layer[..., c] = base[c] * shade * body
    layer[..., 3] = body * 255


def _resolve_controlnet_scales(
    scale: float | list[float] | dict[str, float] | Any,
    *,
    use_seg: bool,
) -> float | list[float]:
    if isinstance(scale, (list, tuple)):
        depth_s, seg_s = float(scale[0]), float(scale[1] if len(scale) > 1 else scale[0])
    elif isinstance(scale, (int, float)):
        depth_s = seg_s = float(scale)
    else:
        depth_s = float(scale.get("depth", 0.55))
        seg_s = float(scale.get("seg", 0.40))
    return [depth_s, seg_s] if use_seg else depth_s


def _prime_snow(
    image: Image.Image,
    weight: np.ndarray,
    *,
    mix: float = 0.55,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    soft = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 2.5)
    soft = np.clip(soft, 0.0, 1.0) * float(mix)
    rng = np.random.default_rng(0)
    patch = rng.random((rgb.shape[0] // 8 + 1, rgb.shape[1] // 8 + 1)).astype(np.float32)
    patch = cv2.resize(patch, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    patch = cv2.GaussianBlur(patch, (0, 0), 3)
    patch = 0.55 + 0.45 * patch
    noise = rng.normal(0.0, 10.0, size=rgb.shape).astype(np.float32)
    snow = np.clip(
        np.stack(
            [
                np.full(rgb.shape[:2], 228.0),
                np.full(rgb.shape[:2], 232.0),
                np.full(rgb.shape[:2], 238.0),
            ],
            axis=-1,
        )
        + noise,
        175,
        250,
    )
    snow = snow * patch[..., None]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    structure = np.clip(gray * 0.55 + 90.0, 70, 220)
    snow = 0.72 * snow + 0.28 * structure[..., None]
    primed = rgb * (1.0 - soft[..., None]) + snow * soft[..., None]
    return Image.fromarray(np.clip(primed, 0, 255).astype(np.uint8))


def _apply_winter_grade(
    image: Image.Image,
    weight: np.ndarray,
    *,
    amount: float = 0.7,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    soft = np.clip(cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 5) * float(amount), 0.0, 1.0)
    amt = float(amount)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green_dom = np.clip((g - np.maximum(r, b) + 8.0) / 35.0, 0.0, 1.0)
    brown = 0.42 * r + 0.38 * g + 0.20 * b
    brown_rgb = np.stack([brown * 1.05, brown * 0.95, brown * 0.85], axis=-1)
    kill = np.clip((0.85 + 0.15 * soft) * np.maximum(green_dom, soft * 0.5) * amt, 0.0, 1.0)
    cooled = rgb * (1.0 - kill[..., None]) + brown_rgb * kill[..., None]
    cooled[..., 0] *= 1.0 - 0.10 * amt
    cooled[..., 1] *= 1.0 - 0.06 * amt
    cooled[..., 2] = np.clip(cooled[..., 2] * (1.0 + 0.04 * amt) + 4.0 * amt, 0, 255)
    gray = cooled.mean(axis=-1, keepdims=True)
    cooled = cooled * (1.0 - 0.35 * amt) + gray * (0.35 * amt)
    rng = np.random.default_rng(1)
    patch = rng.random((rgb.shape[0] // 12 + 1, rgb.shape[1] // 12 + 1)).astype(np.float32)
    patch = cv2.resize(patch, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    patch = cv2.GaussianBlur(patch, (0, 0), 5)
    frost_amt = soft * (0.12 + 0.28 * patch) * amt
    frost = np.array([188.0, 190.0, 196.0], dtype=np.float32)
    frosted = cooled * (1.0 - frost_amt[..., None]) + frost * frost_amt[..., None]
    out = rgb * (1.0 - 0.55 * amt) + frosted * (0.55 * amt)
    out = out * (1.0 - soft[..., None]) + frosted * soft[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _prepare_winter_weight(
    segmentation: SegmentationResult | None,
    width: int,
    height: int,
) -> np.ndarray:
    if segmentation is not None and segmentation.winter_weight is not None:
        weight = segmentation.winter_weight.astype(np.float32)
    else:
        yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        weight = np.clip((yy - 0.25) / 0.75, 0.0, 1.0)
        weight = np.broadcast_to(weight, (height, width)).copy() * 0.4
    if weight.shape[:2] != (height, width):
        weight = cv2.resize(weight, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.GaussianBlur(weight, (0, 0), 2.0), 0.0, 1.0)


def _ensure_same_size(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Resize ``image`` to ``reference`` size when a pipeline ignores width/height."""
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.Resampling.LANCZOS)


def _composite_with_weight(
    original: Image.Image,
    generated: Image.Image,
    weight: np.ndarray,
    *,
    blur_sigma: float = 1.5,
) -> Image.Image:
    generated = _ensure_same_size(generated, original)
    base = np.array(original.convert("RGB"), dtype=np.float32)
    gen = np.array(generated.convert("RGB"), dtype=np.float32)
    h, w = base.shape[:2]
    weight_arr = np.asarray(weight, dtype=np.float32)
    if weight_arr.shape[:2] != (h, w):
        weight_arr = cv2.resize(weight_arr, (w, h), interpolation=cv2.INTER_LINEAR)
    soft = cv2.GaussianBlur(weight_arr, (0, 0), max(blur_sigma, 0.5))
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    out = base * (1.0 - soft) + gen * soft
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _fit_max_side(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None or max(image.size) <= max_side:
        w, h = image.size
        return image.resize(
            (_round_to_multiple(w, 8), _round_to_multiple(h, 8)),
            Image.Resampling.LANCZOS,
        )
    w, h = image.size
    scale = max_side / max(w, h)
    new_w = _round_to_multiple(max(8, int(round(w * scale))), 8)
    new_h = _round_to_multiple(max(8, int(round(h * scale))), 8)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def _resize_rgb(image: np.ndarray, width: int, height: int) -> Image.Image:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return pil
