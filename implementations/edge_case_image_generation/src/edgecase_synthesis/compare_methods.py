"""Side-by-side comparison of three edge-case edit methods (Notebook 1.5).

1. ``inpaint`` — localized hole + SD inpaint (mask required)
2. ``controlnet_dual`` — depth + segmentation ControlNets, full-frame, **no mask**
3. ``instruct`` — instruction image editor (image + text only; no mask / depth / seg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from edgecase_synthesis.generation import (
    GenerationResult,
    _depth_to_control_image,
    _ensure_same_size,
    _fit_for_diffusion,
    _mask_to_pil,
    _composite,
)

COMPARE_METHODS = ("inpaint", "controlnet_dual", "instruct")


@dataclass
class MethodSpec:
    """Human-readable description of one compare method."""

    key: str
    title: str
    uses_mask: bool
    uses_depth: bool
    uses_seg: bool
    summary: str


METHOD_SPECS: dict[str, MethodSpec] = {
    "inpaint": MethodSpec(
        key="inpaint",
        title="Inpaint + mask",
        uses_mask=True,
        uses_depth=True,
        uses_seg=True,
        summary="Edit only inside a mask (ellipse ∩ road ∩ near). Best for local inserts.",
    ),
    "controlnet_dual": MethodSpec(
        key="controlnet_dual",
        title="ControlNet depth + seg (no mask)",
        uses_mask=False,
        uses_depth=True,
        uses_seg=True,
        summary="Full-frame img2img guided by depth and segmentation maps. No hole mask.",
    ),
    "instruct": MethodSpec(
        key="instruct",
        title="Instruction edit (image + prompt)",
        uses_mask=False,
        uses_depth=False,
        uses_seg=False,
        summary="InstructPix2Pix-style editor: only RGB + text instruction. No structure maps.",
    ),
}


@dataclass
class CompareBundle:
    """One image × one anomaly × all methods."""

    sample_name: str
    anomaly_id: str
    prompt: str
    depth: DepthResult
    segmentation: SegmentationResult
    edit_mask: np.ndarray | None
    results: dict[str, GenerationResult] = field(default_factory=dict)


class MethodComparer:
    """Run the three bootcamp compare methods with hardware-selected models."""

    def __init__(
        self,
        *,
        family: str = "sd15",
        base_model_id: str,
        inpaint_model_id: str,
        depth_controlnet_id: str,
        seg_controlnet_id: str,
        instruct_model_id: str,
        vae_id: str | None = None,
        controlnet_scale_depth: float = 0.55,
        controlnet_scale_seg: float = 0.45,
        instruct_image_guidance: float = 1.4,
        instruct_guidance: float = 7.0,
        device: str | None = None,
        seg_as_canny: bool = False,
    ) -> None:
        self.family = str(family).lower()
        self.base_model_id = base_model_id
        self.inpaint_model_id = inpaint_model_id
        self.depth_controlnet_id = depth_controlnet_id
        self.seg_controlnet_id = seg_controlnet_id
        self.instruct_model_id = instruct_model_id
        self.vae_id = vae_id
        self.controlnet_scale_depth = float(controlnet_scale_depth)
        self.controlnet_scale_seg = float(controlnet_scale_seg)
        self.instruct_image_guidance = float(instruct_image_guidance)
        self.instruct_guidance = float(instruct_guidance)
        self.device = resolve_device(device)
        self.seg_as_canny = bool(seg_as_canny)
        self._inpaint_pipe = None
        self._dual_pipe = None
        self._instruct_pipe = None

    def unload(self) -> None:
        self._inpaint_pipe = None
        self._dual_pipe = None
        self._instruct_pipe = None
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

    def _gen(self, seed: int) -> torch.Generator:
        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        return torch.Generator(device=gen_device).manual_seed(int(seed))

    # --- builders ---------------------------------------------------------

    def _build_inpaint(self):
        from diffusers import AutoPipelineForInpainting

        dtype = self._dtype()
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        return self._place(pipe)

    def _build_dual(self):
        from diffusers import ControlNetModel

        dtype = self._dtype()
        depth_cn = ControlNetModel.from_pretrained(self.depth_controlnet_id, torch_dtype=dtype)
        seg_cn = ControlNetModel.from_pretrained(self.seg_controlnet_id, torch_dtype=dtype)
        controlnets = [depth_cn, seg_cn]

        if self.family == "sdxl":
            from diffusers import AutoencoderKL, StableDiffusionXLControlNetImg2ImgPipeline

            kwargs: dict[str, Any] = {"controlnet": controlnets, "torch_dtype": dtype}
            if self.vae_id:
                kwargs["vae"] = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.base_model_id, **kwargs
            )
            return self._place(pipe)

        from diffusers import StableDiffusionControlNetImg2ImgPipeline

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnets,
            torch_dtype=dtype,
            safety_checker=None,
        )
        return self._place(pipe)

    def _build_instruct(self):
        dtype = self._dtype()
        if self.family == "sdxl" and "sdxl" in self.instruct_model_id.lower():
            from diffusers import StableDiffusionXLInstructPix2PixPipeline

            pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                self.instruct_model_id,
                torch_dtype=dtype,
            )
        else:
            from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.instruct_model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return self._place(pipe)

    @property
    def inpaint_pipe(self):
        if self._inpaint_pipe is None:
            self._inpaint_pipe = self._build_inpaint()
        return self._inpaint_pipe

    @property
    def dual_pipe(self):
        if self._dual_pipe is None:
            self._dual_pipe = self._build_dual()
        return self._dual_pipe

    @property
    def instruct_pipe(self):
        if self._instruct_pipe is None:
            self._instruct_pipe = self._build_instruct()
        return self._instruct_pipe

    # --- public API -------------------------------------------------------

    def run_method(
        self,
        method: str,
        image: Image.Image,
        *,
        depth: DepthResult,
        segmentation: SegmentationResult,
        generation_cfg: Any,
        anomaly_cfg: Any,
    ) -> GenerationResult:
        from edgecase_synthesis.config import merge_generation_anomaly

        method = str(method).lower()
        if method not in COMPARE_METHODS:
            raise ValueError(f"Unknown compare method {method!r}. Choose from {COMPARE_METHODS}")

        merged = merge_generation_anomaly(generation_cfg, anomaly_cfg)
        anom = merged.get("anomaly", anomaly_cfg)
        max_side = int(merged.get("max_side", 512))
        seed = int(merged.get("seed", 42))
        prompt = str(merged.get("prompt", ""))
        negative = str(merged.get("negative_prompt", ""))
        anomaly_id = str(anom.get("id", ""))
        family = str(merged.get("family", self.family)).lower()
        original = _fit_for_diffusion(image.convert("RGB"), max_side=max_side, family=family)
        width, height = original.size
        steps = int(merged.get("num_inference_steps", 24))
        guidance = float(merged.get("guidance_scale", 7.5))
        # Inpaint uses high denoise; ControlNet img2img must stay low or it rewrites the scene.
        inpaint_strength = float(merged.get("strength", 0.88))
        cn_strength = float(
            merged.get(
                "controlnet_strength",
                min(0.45, inpaint_strength),
            )
        )

        edit_mask_cfg = dict(anom.get("edit_mask", {"mode": "ellipse"}))
        edit_mask, edit_weight = build_anomaly_edit_mask(
            segmentation,
            edit_mask_cfg,
            width=width,
            height=height,
            depth=depth,
        )

        if method == "inpaint":
            return self._run_inpaint(
                original,
                prompt=prompt,
                negative_prompt=negative,
                steps=steps,
                guidance=guidance,
                strength=min(inpaint_strength, 0.99),
                seed=seed,
                edit_mask=edit_mask,
                edit_mask_cfg=edit_mask_cfg,
                anomaly_id=anomaly_id,
            )
        if method == "controlnet_dual":
            dual_prompt, dual_neg = _dual_fidelity_prompts(prompt, negative, anomaly_id)
            scales = merged.get("controlnet_scale") or {}
            if hasattr(scales, "get"):
                depth_scale = float(scales.get("depth", self.controlnet_scale_depth))
                seg_scale = float(scales.get("seg", self.controlnet_scale_seg))
            else:
                depth_scale = self.controlnet_scale_depth
                seg_scale = self.controlnet_scale_seg
            return self._run_dual(
                original,
                depth,
                segmentation,
                prompt=dual_prompt,
                negative_prompt=dual_neg,
                steps=steps,
                guidance=min(guidance, 6.5),
                strength=float(np.clip(cn_strength, 0.25, 0.60)),
                seed=seed,
                anomaly_id=anomaly_id,
                controlnet_scale_depth=depth_scale,
                controlnet_scale_seg=seg_scale,
            )
        return self._run_instruct(
            original,
            prompt=prompt,
            steps=max(steps, 20),
            seed=seed,
            anomaly_id=anomaly_id,
            edit_mask=None,
            image_guidance=float(
                merged.get("instruct_image_guidance", self.instruct_image_guidance)
            ),
            text_guidance=float(merged.get("instruct_guidance_scale", self.instruct_guidance)),
        )

    def compare_one(
        self,
        image: Image.Image,
        *,
        sample_name: str,
        depth: DepthResult,
        segmentation: SegmentationResult,
        generation_cfg: Any,
        anomaly_cfg: Any,
        methods: tuple[str, ...] = COMPARE_METHODS,
    ) -> CompareBundle:
        from edgecase_synthesis.config import merge_generation_anomaly

        merged = merge_generation_anomaly(generation_cfg, anomaly_cfg)
        anom = merged.get("anomaly", anomaly_cfg)
        anomaly_id = str(anom.get("id", ""))
        prompt = str(merged.get("prompt", ""))
        family = str(merged.get("family", self.family)).lower()
        max_side = int(merged.get("max_side", 512))
        fitted = _fit_for_diffusion(image.convert("RGB"), max_side=max_side, family=family)
        w, h = fitted.size
        edit_mask, _ = build_anomaly_edit_mask(
            segmentation,
            dict(anom.get("edit_mask", {"mode": "ellipse"})),
            width=w,
            height=h,
            depth=depth,
        )
        bundle = CompareBundle(
            sample_name=sample_name,
            anomaly_id=anomaly_id,
            prompt=prompt,
            depth=depth,
            segmentation=segmentation,
            edit_mask=edit_mask,
        )
        for method in methods:
            print(f"  → {method} …", flush=True)
            bundle.results[method] = self.run_method(
                method,
                image,
                depth=depth,
                segmentation=segmentation,
                generation_cfg=generation_cfg,
                anomaly_cfg=anomaly_cfg,
            )
        return bundle

    # --- method implementations -------------------------------------------

    @torch.inference_mode()
    def _run_inpaint(
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
        edit_mask_cfg: dict[str, Any],
        anomaly_id: str,
    ) -> GenerationResult:
        width, height = original.size
        mask_pil = _mask_to_pil(edit_mask)
        if mask_pil.size != (width, height):
            mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        generated = self.inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=original,
            mask_image=mask_pil,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=self._gen(seed),
        ).images[0]
        generated = _ensure_same_size(generated, original)
        blur = float(edit_mask_cfg.get("composite_blur", 0.8))
        output = _composite(original, generated, edit_mask.astype(np.float32), blur_sigma=blur)
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            edit_mask=edit_mask,
            anomaly_id=anomaly_id,
            method="inpaint",
        )

    @torch.inference_mode()
    def _run_dual(
        self,
        original: Image.Image,
        depth: DepthResult,
        segmentation: SegmentationResult,
        *,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
        anomaly_id: str,
        controlnet_scale_depth: float | None = None,
        controlnet_scale_seg: float | None = None,
    ) -> GenerationResult:
        width, height = original.size
        depth_image = _depth_to_control_image(depth, width, height)
        seg_image = _seg_to_control_image(
            segmentation, width, height, as_canny=self.seg_as_canny
        )
        depth_scale = (
            self.controlnet_scale_depth
            if controlnet_scale_depth is None
            else float(controlnet_scale_depth)
        )
        seg_scale = (
            self.controlnet_scale_seg
            if controlnet_scale_seg is None
            else float(controlnet_scale_seg)
        )
        generated = self.dual_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=original,
            control_image=[depth_image, seg_image],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            controlnet_conditioning_scale=[depth_scale, seg_scale],
            generator=self._gen(seed),
        ).images[0]
        generated = _ensure_same_size(generated, original)
        return GenerationResult(
            image=generated,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            edit_mask=None,
            anomaly_id=anomaly_id,
            method="controlnet_dual",
        )

    @torch.inference_mode()
    def _run_instruct(
        self,
        original: Image.Image,
        *,
        prompt: str,
        steps: int,
        seed: int,
        anomaly_id: str,
        edit_mask: np.ndarray | None,
        image_guidance: float | None = None,
        text_guidance: float | None = None,
    ) -> GenerationResult:
        instruction = _to_edit_instruction(prompt, anomaly_id)
        width, height = original.size
        is_sdxl_instruct = "sdxl" in self.instruct_model_id.lower()
        if is_sdxl_instruct:
            # Official SDXL InstructPix2Pix checkpoint expects ~768².
            side = 768
            run_image = original.resize((side, side), Image.Resampling.LANCZOS)
            img_g = 1.5 if image_guidance is None else float(image_guidance)
            txt_g = 3.0 if text_guidance is None else float(text_guidance)
            generated = self.instruct_pipe(
                prompt=instruction,
                image=run_image,
                height=side,
                width=side,
                num_inference_steps=max(steps, 28),
                guidance_scale=txt_g,
                image_guidance_scale=img_g,
                generator=self._gen(seed),
            ).images[0]
        else:
            # SD1.5 InstructPix2Pix — more faithful for photo edits than the SDXL twin.
            target = 512
            if max(width, height) > target:
                scale = target / max(width, height)
                new_w = max(8, int(round(width * scale / 8) * 8))
                new_h = max(8, int(round(height * scale / 8) * 8))
                run_image = original.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                run_image = original
            img_g = (
                self.instruct_image_guidance
                if image_guidance is None
                else float(image_guidance)
            )
            txt_g = self.instruct_guidance if text_guidance is None else float(text_guidance)
            generated = self.instruct_pipe(
                prompt=instruction,
                image=run_image,
                num_inference_steps=steps,
                guidance_scale=txt_g,
                image_guidance_scale=img_g,
                generator=self._gen(seed),
            ).images[0]
        generated = _ensure_same_size(generated, original)
        return GenerationResult(
            image=generated,
            prompt=instruction,
            negative_prompt="",
            seed=seed,
            edit_mask=edit_mask,
            anomaly_id=anomaly_id,
            method="instruct",
        )

    @classmethod
    def from_config(cls, cfg: Any, device: str | None = None) -> "MethodComparer":
        generation = cfg.get("generation", cfg)
        controlnet = generation.get("controlnet", {})
        scales = generation.get("controlnet_scale", {}) or {}
        if device is None:
            hardware = cfg.get("hardware") if hasattr(cfg, "get") else None
            if hardware is not None:
                device = hardware.get("device")
        family = str(generation.get("family", "sd15")).lower()
        seg_cn = str(controlnet.get("seg") or controlnet.get("segmentation") or "")
        if not seg_cn:
            raise ValueError("generation.controlnet.seg required for MethodComparer")
        # SDXL often pairs depth with a Canny ControlNet on the seg map.
        seg_as_canny = "canny" in seg_cn.lower()
        return cls(
            family=family,
            base_model_id=str(generation["base_model_id"]),
            inpaint_model_id=str(generation["inpaint_model_id"]),
            depth_controlnet_id=str(controlnet["depth"]),
            seg_controlnet_id=seg_cn,
            instruct_model_id=str(
                generation.get("instruct_model_id") or "timbrooks/instruct-pix2pix"
            ),
            vae_id=generation.get("vae_id"),
            controlnet_scale_depth=float(scales.get("depth", 0.55)),
            controlnet_scale_seg=float(scales.get("seg", 0.45)),
            instruct_image_guidance=float(generation.get("instruct_image_guidance", 1.4)),
            instruct_guidance=float(generation.get("instruct_guidance_scale", 7.0)),
            device=device,
            seg_as_canny=seg_as_canny,
        )


def _seg_to_control_image(
    segmentation: SegmentationResult,
    width: int,
    height: int,
    *,
    as_canny: bool,
) -> Image.Image:
    colored = np.asarray(segmentation.colored_map, dtype=np.uint8)
    if colored.shape[0] != height or colored.shape[1] != width:
        colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)
    if as_canny:
        gray = cv2.cvtColor(colored, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        rgb = np.stack([edges, edges, edges], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    return Image.fromarray(colored, mode="RGB")


def _dual_fidelity_prompts(
    prompt: str, negative: str, anomaly_id: str
) -> tuple[str, str]:
    """Keep dual ControlNet as a subtle edit of the same photo, not a new scene."""
    fidelity = (
        "same street photograph, identical buildings vehicles lighting and camera angle, "
        "preserve original colors and materials, subtle localized change only, photorealistic"
    )
    dual_prompt = f"{prompt.strip()}, {fidelity}"
    anti_restyle = (
        "different city, new buildings, replaced cars, night scene, stormy sky, "
        "CGI render, oversaturated, painting, watercolor, drastic restyle, "
        "orange color cast, global recolor, fantasy architecture"
    )
    dual_neg = f"{negative.strip()}, {anti_restyle}" if negative.strip() else anti_restyle
    # Fog is global — allow stronger atmosphere but still forbid layout rewrite.
    if anomaly_id == "fog":
        dual_prompt = (
            f"{prompt.strip()}, same cars and road layout, same camera, "
            "only weather/atmosphere changes, photorealistic"
        )
    return dual_prompt, dual_neg


def _to_edit_instruction(prompt: str, anomaly_id: str) -> str:
    """Turn a descriptive generation prompt into an InstructPix2Pix instruction."""
    mapping = {
        "pothole": (
            "Add a realistic deep pothole in the foreground road asphalt, "
            "without changing the buildings, cars, sky, or camera view"
        ),
        "traffic_cone": (
            "Add one bright orange traffic cone with a white stripe standing on the road, "
            "without recoloring the car or buildings or changing the rest of the scene"
        ),
        "ground_animal": (
            "Add a realistic animal standing on the road ahead, "
            "without changing the background or camera view"
        ),
        "fog": (
            "Add dense realistic fog and haze over the street, reducing distant visibility, "
            "without replacing the cars or road layout"
        ),
    }
    if anomaly_id in mapping:
        return mapping[anomaly_id]
    # Fallback: shorten descriptive prompt into an instruction.
    short = prompt.strip().split(",")[0].strip()
    if short.lower().startswith("photorealistic"):
        short = short[len("photorealistic") :].strip()
    return f"Edit the photo carefully: {short}, keep the rest of the scene unchanged"
