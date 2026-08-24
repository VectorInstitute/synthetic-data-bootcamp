"""Side-by-side comparison of edge-case edit / generate methods (Notebook 1.5).

NB1.5 compares five edit paths; **production (NB1 / NB2) defaults to ``instruct`` (Klein)
+ API VLM judge** — see ``configs/default/generation.yaml`` and ``judge.yaml``.

1. ``controlnet_dual`` — depth + segmentation ControlNets, full-frame
2. ``inpaint`` — localized hole + SD/Klein inpaint (mask required)
3. ``instruct`` — Klein / IP2P instruction editor (**small local**; default generator)
4. ``vlm_generate_local`` — Qwen-Image-Edit (**large local** instruct-class editor)
5. ``vlm_generate_api`` — cloud Gemini / GPT Image API edit
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

# Diffusion stack (offline once weights are cached).
LOCAL_DIFFUSION_METHODS = ("controlnet_dual", "inpaint", "instruct")
# Large local + cloud instruction-class editors (NB1.5 only; not the NB1/NB2 default).
VLM_GENERATE_METHODS = ("vlm_generate_local", "vlm_generate_api")
# Shorthand resolved via generation.vlm_generate_backend (local | api).
VLM_GENERATE_ALIAS = "vlm_generate"
# NB1 / NB2 production methods (Klein ``instruct`` is the default generator).
PIPELINE_METHODS = LOCAL_DIFFUSION_METHODS
# Default NB1.5 grid — all five methods.
COMPARE_METHODS = LOCAL_DIFFUSION_METHODS + VLM_GENERATE_METHODS
ALL_COMPARE_METHODS = COMPARE_METHODS + (VLM_GENERATE_ALIAS,)


def resolve_effective_method(method: str, generation_cfg: Any) -> str:
    """Map ``vlm_generate`` alias → local or API backend from config."""
    method = str(method).lower()
    if method != VLM_GENERATE_ALIAS:
        return method
    backend = str(generation_cfg.get("vlm_generate_backend", "local")).lower()
    return "vlm_generate_local" if backend == "local" else "vlm_generate_api"


def prompt_method_key(method: str) -> str:
    """YAML ``methods`` block key (vlm_generate_* share ``vlm_generate`` fallbacks)."""
    if method in {"vlm_generate_local", "vlm_generate_api", VLM_GENERATE_ALIAS}:
        return "vlm_generate"
    return method


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
        summary=(
            "Local inserts (cone, debris): masked inpaint. L4 uses FLUX.2-klein-4B; "
            "CPU uses SD1.5. Rest of the photo stays outside the mask."
        ),
    ),
    "controlnet_dual": MethodSpec(
        key="controlnet_dual",
        title="ControlNet (depth + seg)",
        uses_mask=False,
        uses_depth=True,
        uses_seg=True,
        summary=(
            "Full-frame img2img locked by depth+seg. Best for mild global weather "
            "(fog) at low strength — not for placing a small object."
        ),
    ),
    "instruct": MethodSpec(
        key="instruct",
        title="Klein instruct (small local)",
        uses_mask=False,
        uses_depth=False,
        uses_seg=False,
        summary=(
            "Small local instruction editor (FLUX.2-klein-4B on L4; InstructPix2Pix on CPU). "
            "**Default generator for NB1 / NB2.** RGB + text; weak spatial control for tiny inserts."
        ),
    ),
    "vlm_generate_local": MethodSpec(
        key="vlm_generate_local",
        title="Qwen-Image-Edit (large local)",
        uses_mask=False,
        uses_depth=False,
        uses_seg=False,
        summary=(
            "Large local instruction-class editor (~20B Qwen-Image-Edit). Same role as Klein "
            "but heavier — NB1.5 comparison only. gpu_l4x2 recommended."
        ),
    ),
    "vlm_generate_api": MethodSpec(
        key="vlm_generate_api",
        title="VLM edit (API)",
        uses_mask=False,
        uses_depth=False,
        uses_seg=False,
        summary=(
            "Cloud image model (Gemini *-image / GPT Image). Stronger semantics; "
            "NB1.5 comparison column — production pipeline uses API for **judge**, not edit."
        ),
    ),
    VLM_GENERATE_ALIAS: MethodSpec(
        key=VLM_GENERATE_ALIAS,
        title="VLM edit (config backend)",
        uses_mask=False,
        uses_depth=False,
        uses_seg=False,
        summary=(
            "Resolves to Qwen-Image-Edit or API via ``generation.vlm_generate_backend``. "
            "Advanced alias — production batch uses ``instruct`` (Klein) instead."
        ),
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
        instruct_num_inference_steps: int | None = None,
        inpaint_num_inference_steps: int | None = None,
        inpaint_guidance_scale: float | None = None,
        device: str | None = None,
        seg_as_canny: bool = False,
        vlm_api_model: str = "gemini-3.1-flash-image",
        vlm_generate_backend: str = "local",
        vlm_mode: str = "edit",
        vlm_provider: str | None = None,
        vlm_api_key: str | None = None,
        vlm_max_side: int = 1024,
        vlm_size: str = "1024x1024",
        vlm_local_model_id: str = "Qwen/Qwen-Image-Edit",
        vlm_local_num_inference_steps: int = 20,
        vlm_local_true_cfg_scale: float = 4.0,
        vlm_local_max_side: int = 768,
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
        self.instruct_num_inference_steps = instruct_num_inference_steps
        self.inpaint_num_inference_steps = inpaint_num_inference_steps
        self.inpaint_guidance_scale = inpaint_guidance_scale
        self.device = resolve_device(device)
        self.seg_as_canny = bool(seg_as_canny)
        self.vlm_api_model = str(vlm_api_model)
        self.vlm_generate_backend = str(vlm_generate_backend).lower()
        self.vlm_mode = str(vlm_mode).lower()
        self.vlm_provider = vlm_provider
        self.vlm_api_key = vlm_api_key
        self.vlm_max_side = int(vlm_max_side)
        self.vlm_size = str(vlm_size)
        self.vlm_local_model_id = str(vlm_local_model_id)
        self.vlm_local_num_inference_steps = int(vlm_local_num_inference_steps)
        self.vlm_local_true_cfg_scale = float(vlm_local_true_cfg_scale)
        self.vlm_local_max_side = int(vlm_local_max_side)
        self._inpaint_pipe = None
        self._dual_pipe = None
        self._instruct_pipe = None

    @staticmethod
    def _is_klein_model(model_id: str) -> bool:
        mid = str(model_id).lower()
        return "klein" in mid or "flux.2" in mid or "flux2" in mid

    @property
    def instruct_is_klein(self) -> bool:
        return self._is_klein_model(self.instruct_model_id)

    @property
    def inpaint_is_klein(self) -> bool:
        return self._is_klein_model(self.inpaint_model_id)

    @property
    def uses_klein(self) -> bool:
        return self.instruct_is_klein or self.inpaint_is_klein

    def unload(self) -> None:
        from edgecase_synthesis.vlm_edit_local import unload_qwen_edit_pipeline

        self._inpaint_pipe = None
        self._dual_pipe = None
        self._instruct_pipe = None
        unload_qwen_edit_pipeline()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dtype(self, *, for_klein: bool = False) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if for_klein and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _free_other_edit_pipes(self, *, keep: str) -> None:
        """Drop sibling edit pipes so Klein + SD ControlNet do not share the L4."""
        if keep != "inpaint":
            self._inpaint_pipe = None
        if keep != "dual":
            self._dual_pipe = None
        if keep != "instruct":
            self._instruct_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    @staticmethod
    def _fit_klein_image(image: Image.Image, *, max_side: int = 768) -> Image.Image:
        """Resize to multiples of 16; cap long side for VRAM."""
        width, height = image.size
        long = max(width, height)
        target = min(long, max_side)
        scale = target / long if long else 1.0
        new_w = max(16, int(round(width * scale / 16) * 16))
        new_h = max(16, int(round(height * scale / 16) * 16))
        if (new_w, new_h) == (width, height):
            return image
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # --- builders ---------------------------------------------------------

    def _build_inpaint(self):
        if self.inpaint_is_klein:
            self._free_other_edit_pipes(keep="inpaint")
            from diffusers import Flux2KleinInpaintPipeline

            pipe = Flux2KleinInpaintPipeline.from_pretrained(
                self.inpaint_model_id,
                torch_dtype=self._dtype(for_klein=True),
            )
            return self._place(pipe)

        from diffusers import AutoPipelineForInpainting

        dtype = self._dtype()
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        return self._place(pipe)

    def _build_dual(self):
        if self.uses_klein:
            self._free_other_edit_pipes(keep="dual")
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
        if self.instruct_is_klein:
            self._free_other_edit_pipes(keep="instruct")
            from diffusers import Flux2KleinPipeline

            pipe = Flux2KleinPipeline.from_pretrained(
                self.instruct_model_id,
                torch_dtype=self._dtype(for_klein=True),
            )
            return self._place(pipe)
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
        seed_offset: int = 0,
    ) -> GenerationResult:
        from edgecase_synthesis.config import merge_generation_anomaly, resolve_method_prompt

        method = str(method).lower()
        if method not in ALL_COMPARE_METHODS:
            raise ValueError(f"Unknown compare method {method!r}. Choose from {ALL_COMPARE_METHODS}")

        effective = resolve_effective_method(method, generation_cfg)
        prompt_key = prompt_method_key(effective)
        merged = merge_generation_anomaly(generation_cfg, anomaly_cfg, method=prompt_key)
        anom = merged.get("anomaly", anomaly_cfg)
        max_side = int(merged.get("max_side", 512))
        seed = int(merged.get("seed", 42)) + int(seed_offset)
        prompt, negative = resolve_method_prompt(merged, prompt_key)
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
        padding_crop = merged.get("padding_mask_crop", None)
        padding_crop = int(padding_crop) if padding_crop not in (None, "", False) else None

        if effective == "vlm_generate_api":
            return self._run_vlm_generate_api(
                original,
                prompt=prompt,
                seed=seed,
                anomaly_id=anomaly_id,
                generation_cfg=merged,
            )
        if effective == "vlm_generate_local":
            return self._run_vlm_edit_local(
                original,
                prompt=prompt,
                seed=seed,
                anomaly_id=anomaly_id,
                generation_cfg=merged,
                family=family,
            )

        if effective == "inpaint":
            if self.inpaint_is_klein:
                klein_steps = merged.get("inpaint_num_inference_steps", self.inpaint_num_inference_steps)
                steps = int(klein_steps) if klein_steps not in (None, "") else steps
                klein_gs = merged.get("inpaint_guidance_scale", self.inpaint_guidance_scale)
                guidance = float(klein_gs) if klein_gs not in (None, "") else guidance
                run_strength = min(float(merged.get("strength", 1.0)), 1.0)
            else:
                run_strength = min(inpaint_strength, 0.99)
            return self._run_inpaint(
                original,
                prompt=prompt,
                negative_prompt=negative,
                steps=steps,
                guidance=guidance,
                strength=run_strength,
                seed=seed,
                edit_mask=edit_mask,
                edit_weight=edit_weight,
                edit_mask_cfg=edit_mask_cfg,
                anomaly_id=anomaly_id,
                padding_mask_crop=padding_crop,
            )
        if effective == "controlnet_dual":
            scales = merged.get("controlnet_scale") or {}
            if hasattr(scales, "get"):
                depth_scale = float(scales.get("depth", self.controlnet_scale_depth))
                seg_scale = float(scales.get("seg", self.controlnet_scale_seg))
            else:
                depth_scale = self.controlnet_scale_depth
                seg_scale = self.controlnet_scale_seg
            local_ids = {
                str(x) for x in (merged.get("local_anomaly_ids") or [])
            }
            global_ids = {
                str(x) for x in (merged.get("global_anomaly_ids") or [])
            }
            if anomaly_id in global_ids:
                strength_cap = float(merged.get("controlnet_strength_cap_global", 0.70))
            elif anomaly_id in local_ids:
                strength_cap = float(merged.get("controlnet_strength_cap_local", 0.45))
            else:
                strength_cap = float(merged.get("controlnet_strength_cap_default", 0.50))
            return self._run_dual(
                original,
                depth,
                segmentation,
                prompt=prompt,
                negative_prompt=negative,
                steps=steps,
                guidance=min(guidance, 6.5),
                strength=float(np.clip(cn_strength, 0.25, strength_cap)),
                seed=seed,
                anomaly_id=anomaly_id,
                controlnet_scale_depth=depth_scale,
                controlnet_scale_seg=seg_scale,
            )
        instruct_steps = merged.get("instruct_num_inference_steps", self.instruct_num_inference_steps)
        return self._run_instruct(
            original,
            prompt=prompt,
            steps=steps,
            seed=seed,
            anomaly_id=anomaly_id,
            edit_mask=None,
            image_guidance=float(
                merged.get("instruct_image_guidance", self.instruct_image_guidance)
            ),
            text_guidance=float(merged.get("instruct_guidance_scale", self.instruct_guidance)),
            instruct_steps=instruct_steps,
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
            # Heavy models (Klein, Qwen-Image-Edit, SD ControlNet) do not share one L4.
            if self.device.type == "cuda":
                self.unload()
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
        edit_weight: np.ndarray,
        edit_mask_cfg: dict[str, Any],
        anomaly_id: str,
        padding_mask_crop: int | None = None,
    ) -> GenerationResult:
        if self.inpaint_is_klein:
            run_image = self._fit_klein_image(original, max_side=768)
            rw, rh = run_image.size
            # Rebuild mask at Klein resolution (ellipse was built at `original` size).
            mask_arr = edit_mask
            if mask_arr.shape[0] != rh or mask_arr.shape[1] != rw:
                mask_arr = cv2.resize(
                    mask_arr.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST
                )
            mask_pil = _mask_to_pil(mask_arr)
            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "image": run_image,
                "mask_image": mask_pil,
                "height": rh,
                "width": rw,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "strength": float(strength),
                "generator": self._gen(seed),
            }
            # Flux2KleinInpaintPipeline has no negative_prompt arg.
            if padding_mask_crop is not None and padding_mask_crop > 0:
                kwargs["padding_mask_crop"] = int(padding_mask_crop)
            try:
                generated = self.inpaint_pipe(**kwargs).images[0]
            except TypeError:
                kwargs.pop("padding_mask_crop", None)
                generated = self.inpaint_pipe(**kwargs).images[0]
            generated = _ensure_same_size(generated, original)
            # Map edit_mask back to original size for annotation/viz.
            out_mask = edit_mask
        else:
            width, height = original.size
            mask_pil = _mask_to_pil(edit_mask)
            if mask_pil.size != (width, height):
                mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "image": original,
                "mask_image": mask_pil,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "strength": strength,
                "generator": self._gen(seed),
            }
            if padding_mask_crop is not None and padding_mask_crop > 0:
                kwargs["padding_mask_crop"] = int(padding_mask_crop)
            try:
                generated = self.inpaint_pipe(**kwargs).images[0]
            except TypeError:
                kwargs.pop("padding_mask_crop", None)
                generated = self.inpaint_pipe(**kwargs).images[0]
            generated = _ensure_same_size(generated, original)
            out_mask = edit_mask

        # Dedicated inpaint checkpoints already keep pixels outside the mask.
        recomposite = bool(edit_mask_cfg.get("recomposite", False))
        if recomposite:
            blur = float(edit_mask_cfg.get("composite_blur", 1.0))
            output = _composite(original, generated, edit_weight, blur_sigma=blur)
        else:
            output = generated
        return GenerationResult(
            image=output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            edit_mask=out_mask,
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
        instruct_steps: int | None = None,
        result_method: str = "instruct",
    ) -> GenerationResult:
        instruction = str(prompt).strip()
        width, height = original.size

        if self.instruct_is_klein:
            # Distilled Klein: ~4 steps when configured; guidance ignored when >1.
            n_steps = int(instruct_steps) if instruct_steps not in (None, "") else 4
            txt_g = 1.0 if text_guidance is None else float(text_guidance)
            run_image = self._fit_klein_image(original, max_side=768)
            generated = self.instruct_pipe(
                prompt=instruction,
                image=run_image,
                height=run_image.size[1],
                width=run_image.size[0],
                num_inference_steps=n_steps,
                guidance_scale=txt_g,
                generator=self._gen(seed),
            ).images[0]
        elif "sdxl" in self.instruct_model_id.lower():
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
            # SD1.5 InstructPix2Pix — CPU / fallback path.
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
            n_steps = int(instruct_steps) if instruct_steps not in (None, "") else max(steps, 20)
            generated = self.instruct_pipe(
                prompt=instruction,
                image=run_image,
                num_inference_steps=n_steps,
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
            method=result_method,
        )

    def _run_vlm_edit_local(
        self,
        original: Image.Image,
        *,
        prompt: str,
        seed: int,
        anomaly_id: str,
        generation_cfg: Any,
        family: str,
    ) -> GenerationResult:
        from edgecase_synthesis.vlm_edit_local import VlmLocalEditConfig, edit_with_qwen_local

        self._free_other_edit_pipes(keep="")
        cfg = VlmLocalEditConfig(
            model_id=str(generation_cfg.get("vlm_local_model_id") or self.vlm_local_model_id),
            num_inference_steps=int(
                generation_cfg.get("vlm_local_num_inference_steps", self.vlm_local_num_inference_steps)
            ),
            true_cfg_scale=float(
                generation_cfg.get("vlm_local_true_cfg_scale", self.vlm_local_true_cfg_scale)
            ),
            max_side=int(generation_cfg.get("vlm_local_max_side", self.vlm_local_max_side)),
        )
        edited = edit_with_qwen_local(
            original,
            prompt,
            config=cfg,
            device=str(self.device),
            seed=seed,
            family=family,
        )
        return GenerationResult(
            image=edited,
            prompt=prompt,
            negative_prompt="",
            seed=seed,
            edit_mask=None,
            anomaly_id=anomaly_id,
            method="vlm_generate_local",
        )

    def _run_vlm_generate_api(
        self,
        original: Image.Image,
        *,
        prompt: str,
        seed: int,
        anomaly_id: str,
        generation_cfg: Any,
    ) -> GenerationResult:
        from edgecase_synthesis.vlm_generate import VlmGenerateConfig, generate_with_vlm

        model = str(generation_cfg.get("vlm_api_model", self.vlm_api_model))
        mode = str(generation_cfg.get("vlm_mode", self.vlm_mode)).lower()
        provider = generation_cfg.get("vlm_provider", self.vlm_provider)
        api_key = generation_cfg.get("vlm_api_key", self.vlm_api_key)
        max_side = int(generation_cfg.get("vlm_max_side", self.vlm_max_side))
        size = str(generation_cfg.get("vlm_size", self.vlm_size))
        aspect = generation_cfg.get("vlm_aspect_ratio")

        cfg = VlmGenerateConfig(
            model=model,
            mode="generate" if mode == "generate" else "edit",
            provider=provider,  # type: ignore[arg-type]
            api_key=api_key,
            aspect_ratio=str(aspect) if aspect not in (None, "") else None,
            size=size,
            max_side=max_side,
        )
        seed_image = None if cfg.mode == "generate" else original
        generated = generate_with_vlm(prompt, seed_image=seed_image, config=cfg)
        if seed_image is not None:
            generated = _ensure_same_size(generated, original)
        return GenerationResult(
            image=generated,
            prompt=prompt,
            negative_prompt="",
            seed=seed,
            edit_mask=None,
            anomaly_id=anomaly_id,
            method="vlm_generate_api",
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
        instruct_steps = generation.get("instruct_num_inference_steps", None)
        inpaint_steps = generation.get("inpaint_num_inference_steps", None)
        inpaint_gs = generation.get("inpaint_guidance_scale", None)
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
            instruct_num_inference_steps=(
                int(instruct_steps) if instruct_steps not in (None, "") else None
            ),
            inpaint_num_inference_steps=(
                int(inpaint_steps) if inpaint_steps not in (None, "") else None
            ),
            inpaint_guidance_scale=(
                float(inpaint_gs) if inpaint_gs not in (None, "") else None
            ),
            device=device,
            seg_as_canny=seg_as_canny,
            vlm_api_model=str(generation.get("vlm_api_model") or "gemini-3.1-flash-image"),
            vlm_generate_backend=str(generation.get("vlm_generate_backend") or "local"),
            vlm_mode=str(generation.get("vlm_mode") or "edit"),
            vlm_provider=generation.get("vlm_provider"),
            vlm_api_key=generation.get("vlm_api_key"),
            vlm_max_side=int(generation.get("vlm_max_side") or 1024),
            vlm_size=str(generation.get("vlm_size") or "1024x1024"),
            vlm_local_model_id=str(generation.get("vlm_local_model_id") or "Qwen/Qwen-Image-Edit"),
            vlm_local_num_inference_steps=int(generation.get("vlm_local_num_inference_steps") or 20),
            vlm_local_true_cfg_scale=float(generation.get("vlm_local_true_cfg_scale") or 4.0),
            vlm_local_max_side=int(generation.get("vlm_local_max_side") or 768),
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
