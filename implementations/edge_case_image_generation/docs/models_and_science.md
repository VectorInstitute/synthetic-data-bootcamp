# Models, ControlNet, and the Science Behind Notebook 1

This guide explains **what we are using today**, **why**, **what the knobs mean**, and **what you could swap in** at different compute budgets. It matches the code under `src/edgecase_synthesis/` and the configs in `configs/`.

---

## 1. Pipeline at a glance

```text
real photo
   │
   ├─► Depth Anything V2  ──► depth map ────────┐
   │                                              │
   ├─► Seg / RailSem19 GT ─► ADE20K colors ─────┼─► MultiControlNet (depth + seg)
   │                         + edit mask         │         │
   │                                              │         ▼
   └──────────────────────────────────────────────┴─► SD 1.5 img2img (anomaly prompt)
                                                        │
                                                        ▼
                                              soft composite (lock outside mask)
                                                        │
                                                        ▼
                                        OWL-ViT boxes + MobileSAM masks
                                                        │
                                                        ▼
                                   (later) VLM judge → accept / retry / reject
```

**Anomaly configs (plug-and-play):** shared knobs in `configs/generation/default.yaml`;
per-anomaly prompts / edit masks / optional overrides in
`configs/generation/anomalies/<dataset>/<anomaly>.yaml`
(e.g. `railsem19/traffic_cone.yaml`). Notebook 1 workshop list:
cone, fallen branch, deer, dog. Weather (`snow_on_rails`) remains on disk.

**Design principle:** do not invent a new railway scene from text. Start from a real cab-view photo, lock geometry with depth + ADE20K seg, inject a rare attribute (cone, branch, animal, …), then label it.

That is why we use **img2img + MultiControlNet**, not text-to-image alone.

---

## 2. Stable Diffusion 1.5 — how it works

### 2.1 Latent diffusion in one paragraph

SD 1.5 is a **latent diffusion model**:

1. An **autoencoder (VAE)** compresses an RGB image \(x\) into a lower-res latent \(z\) (~8× spatially smaller).
2. A **U-Net** is trained to predict the noise that was added to \(z\) at timestep \(t\).
3. A **text encoder (CLIP ViT-L/14)** turns the prompt into embeddings that **cross-attend** into the U-Net, steering what content appears.
4. At inference we start from noise (or a noised init image) and iteratively **denoise** for \(T\) steps.

Working in latent space is why SD 1.5 fits on modest GPUs: most compute is on ~64×64 latents for a 512×512 image (we often run up to `max_side: 768`).

### 2.2 Modes we care about

| Mode | Init | What it means here |
|------|------|--------------------|
| **txt2img** | pure noise | Can invent wrong camera / track layout — avoided |
| **img2img** | encode real photo, add noise by `strength` | Keeps layout *somewhat*; prompt can still wander |
| **inpaint** | special UNet + mask | Needs the *inpaint* checkpoint; plain SD 1.5 + ControlNet-Inpaint **silently ignores** the mask |

Our path: **ControlNet depth + img2img**, then **mask composite** in numpy so pixels outside the snow region stay the (winter-graded) photo.

### 2.3 Hyperparameters in `configs/generation/default.yaml`

| Knob | Typical range | What it does | Our default |
|------|---------------|--------------|-------------|
| `num_inference_steps` | 15–50 | More steps → cleaner denoising, slower | **28** |
| `guidance_scale` (CFG) | 5–12 | How hard to follow the prompt vs stay natural. Too high → harsh / oversaturated | **8.0** |
| `strength` (img2img) | 0.3–0.85 | Fraction of the denoising schedule that rewrites the init. Low = stay close to photo; high = freer (and easier to break rails) | **0.55** |
| `controlnet_scale` | 0.3–1.0 | How strongly depth ControlNet locks geometry. High = stick to depth; low = prompt wins more | **0.55** |
| `seed` | int | Reproducibility | **45** |
| `max_side` | 512–1024 | Long-side resize before diffusion (VRAM / speed) | **768** |
| `snow_prime` | 0–1 | How hard we bias track pixels toward snow **before** diffusion (engineering, not SD) | **0.45** |
| `winter_grade` | 0–1 | How hard we desaturate/cool vegetation (engineering) | **0.9** |
| `prompt` / `negative_prompt` | text | Positive steers content; negative reduces common failure modes | see yaml |

**Intuition for `strength`:** if strength = 0.55 and steps = 28, the model effectively starts ~halfway through the noise schedule on the encoded photo, then takes the remaining steps. It is *not* “55% opacity blend.”

**Why priming exists:** without brightening track pixels toward snow, img2img often stays stuck on wet dark ballast — the posterior is closer to “keep dark gravel” than “invent snow.” Priming shifts the init so snow is an easier continuation.

### 2.4 Diffusion base — options & compute

| Model | Params (approx) | VRAM (fp16, ~768) | Notes |
|-------|-----------------|-------------------|-------|
| **SD 1.5** ← *we use* | ~0.9B UNet + VAE + CLIP | **CPU:** slow but works · **8GB:** comfortable · **24GB:** easy | Best laptop default; ControlNet ecosystem is mature |
| SD 2.1 | similar | similar | Different tokenizer/resolution; less ecosystem match for classic ControlNets |
| **SDXL** base | ~2.6B | **8GB:** tight / need offload · **12–16GB:** OK · **24GB:** comfortable | Better detail; needs SDXL ControlNets |
| **SDXL-Turbo** | SDXL-sized | similar to SDXL | Few-step; great for batch; less precise structure unless ControlNeted |
| **FLUX.1-schnell** | ~12B | **24GB:** possible with care · **40GB+:** comfortable | Project “production” target in `docs/project_summary.md`; not laptop-friendly |
| SD 1.5 **inpaint** checkpoint | same size | same as SD 1.5 | Use this *if* you switch to a true inpaint pipeline |

Apple **MPS** (your machine): SD 1.5 + ControlNet works; expect slower than CUDA and occasional float32 preference.

---

## 3. ControlNet — what it is and what it does

### 3.1 The problem ControlNet solves

Plain SD follows the **text prompt** but is free to change camera angle, invent poles, warp rails, etc. For safety-critical synthesis we need: *same scene geometry, different rare attribute*.

**ControlNet** (Zhang et al., 2023) attaches a trainable copy of the U-Net encoder that consumes a **spatial condition** (depth map, edges, pose, segmentation palette, …) and injects features into the frozen SD U-Net via zero-initialized convolutions. At inference:

\[
\text{UNet}(z_t, t, \text{text},\; \underbrace{\text{ControlNet}(c)}_{\text{spatial lock}})
\]

So the prompt says *what* (snowy day), ControlNet says *where structure lives* (ground plane, poles, track corridor depth).

### 3.2 What we feed it

| Condition | Model / source | Used now? | Role |
|-----------|----------------|-----------|------|
| **Depth** | Depth Anything V2 → Inferno colormap | **Yes** | Preserve 3D layout / perspective |
| **Seg** | ADE20K palette / RS19→ADE colors | **Yes** (MultiControlNet) | Region layout; must use ADE20K colors |
| **Canny / HED** | edge detectors | No | Strong line lock; can fight inserts |
| **OpenPose** | pose | No | People, not rails |

We run **depth + seg MultiControlNet**. Segmentation also builds **edit masks** from anomaly YAML (`blob` / `strip` / `snow`). MobileSAM is for **annotation only** — its random colors must not feed ControlNet-seg.

### 3.3 `controlnet_scale`

Scales the ControlNet residual before it is added to the U-Net. Think of it as a soft dial:

- **Too high** → image looks rigidly glued to depth; hard to change appearance
- **Too low** → depth ignored; floating objects / bent rails return
- **~0.5–0.7** → usual sweet spot for img2img edits

### 3.4 ControlNet options & compute

ControlNets are **extra** encoder copies (~1.2B for SD 1.5 depth). Memory ≈ base SD + ControlNet (+ second ControlNet if Multi).

| Condition type | HF id (SD 1.5) | When to use |
|----------------|----------------|-------------|
| Depth | `lllyasviel/sd-controlnet-depth` ← *ours* | Layout lock for outdoor scenes |
| Seg | `lllyasviel/sd-controlnet-seg` | Needs **ADE20K palette** colors — random SAM colors break it |
| Canny | `lllyasviel/sd-controlnet-canny` | Strong edges; good for hard structure |
| SoftEdge / MLSD | various | Lines / architecture |

For **SDXL**, use SDXL-specific ControlNets (different weights). Do not mix SD 1.5 ControlNet with SDXL UNet.

---

## 4. Depth Anything V2

### 4.1 What it estimates

**Monocular relative depth:** for each pixel, a scalar that orders nearer vs farther in the scene. It is **not** metric depth in meters (unless separately calibrated).

We normalize to \([0,1]\) and render a colormap for ControlNet. Thin rails coplanar with ballast usually **do not** appear as separate depth ridges — that is expected. ControlNet needs the ground plane and major structures, not millimetre rail profiles.

### 4.2 Why Base

| Variant | Size | Quality | Compute |
|---------|------|---------|---------|
| Small | lightest | softer rails/poles | **CPU OK** · **8GB trivial** |
| **Base** ← *we use* | medium | sharper structure | **CPU OK (slow)** · **8GB easy** · **MPS fine** |
| Large | heaviest | best monocular depth | **8GB OK** · **24GB easy** |

HF id: `depth-anything/Depth-Anything-V2-Base-hf`

### 4.3 Alternatives

| Model | Notes | Compute |
|-------|-------|---------|
| Depth Anything V2 Small/Large | Same family, scale knob | see above |
| MiDaS / DPT | Older; still fine for ControlNet | similar |
| ZoeDepth | Metric depth heads | more VRAM |
| Stereo / LiDAR | Real metric; out of scope for this photo pipeline | hardware |

---

## 5. Segmentation (SegFormer + RailSem19 GT)

### 5.1 Role in *this* repo

Two jobs:

1. **Edit weights** — which pixels get snow priming / composite (`snow_weight`), which get winter grade (`winter_weight`).
2. **ADE20K-colored map** — ready if we re-enable seg ControlNet later.

For RailSem19 samples we prefer **GT uint8 labels** (`find_railsem19_label_map`) over SegFormer predictions — better track / trackbed boundaries.

### 5.2 Why not MobileSAM for ControlNet-seg

MobileSAM (and classic SAM) produce **instance** masks with **arbitrary colors**. `sd-controlnet-seg` was trained on the **ADE20K color protocol**. Feeding random colors is like speaking the wrong language — we saw broken geometry on the right side of edits. SegFormer-B0 ADE20K speaks the right color language.

### 5.3 Options & compute

| Model | Notes | Compute |
|-------|-------|---------|
| **SegFormer-B0 ADE** ← *fallback* | Tiny, ADE colors | **CPU/8GB fine** |
| SegFormer-B2/B5 | Better accuracy | **8–12GB** comfortable |
| Mask2Former ADE | Stronger semantic | **12–24GB** |
| OneFormer | Multi-task | heavier |
| RailSem19 GT | Best for this dataset | disk only |
| SAM 2 / MobileSAM | Great **instance** masks for annotation; bad as ControlNet-seg colors unless remapped | MobileSAM: **CPU/8GB** · SAM2-L: **12–24GB** |

Config: `nvidia/segformer-b0-finetuned-ade-512-512`

---

## 6. Annotation: OWL-ViT + MobileSAM

### 6.1 What they do

1. **OWL-ViT** — open-vocabulary detector: text queries → boxes (“snow on railway track”, “tree”, …).
2. **MobileSAM** — given a box, predicts a pixel mask.

Together they are a **lightweight stand-in** for **Grounded-SAM 2** (production target in the project summary).

### 6.2 Limits (know them)

- OWL-ViT struggles with abstract weather (“snow”) unless the visual evidence is strong — hence concrete phrases in `configs/annotation/default.yaml`.
- MobileSAM is smaller/faster than SAM 2; masks are coarser on thin rails.
- Annotation quality ≠ generation quality; the VLM judge (later) scores both image and labels.

### 6.3 Options & compute

| Stack | Notes | Compute |
|-------|-------|---------|
| **OWL-ViT-B/32 + MobileSAM** ← *now* | Local, light | **CPU workable** · **8GB easy** |
| OWL-ViT-L/14 | Better boxes | **8–12GB** |
| Grounding DINO + SAM | Strong open-vocab | **12–24GB** |
| **Grounded-SAM 2** | Project target | **16–24GB** typical |
| YOLO-World / RF-DETR | Faster detectors | **8–16GB** |

---

## 7. Engineering around the models (not “magic”)

These are deliberate hacks because **SD 1.5 alone is weak at localized weather edits**:

| Step | Why |
|------|-----|
| **Winter grade** | Diffusion only rewrites soft-masked track pixels; without grading, hills stay summer green → uncanny |
| **Snow prime** | Biases latents toward “snow continuation” so img2img doesn’t preserve wet ballast |
| **Soft snow_weight** | Rail ribbons + light ballast dusting — avoids a white slab between dual tracks |
| **Soft composite** | Guarantees poles/sky/layout outside the weight map stay photographic |

If you upgrade to FLUX/SDXL with stronger prompting, you may dial these down — but structure lock (ControlNet or equivalent) remains essential.

---

## 8. End-to-end VRAM cheat sheet

Rough **peak** usage for *one* forward of each stage (fp16, ~768 long side). Numbers are order-of-magnitude.

| Stage | CPU | 8GB GPU | 24GB GPU |
|-------|-----|---------|----------|
| Depth V2 Base | OK (slow) | easy | easy |
| SegFormer-B0 | OK | easy | easy |
| SD 1.5 + 1× ControlNet | possible, very slow | **sweet spot** | easy; try SDXL |
| SDXL + ControlNet | painful | offload / low res | comfortable |
| FLUX.1-schnell | no | no / extreme offload | yes with care |
| OWL-ViT + MobileSAM | OK | easy | easy |
| Qwen2.5-VL-7B judge (planned) | no | quantized maybe | **comfortable** |
| YOLO train (Notebook 3) | tiny models only | YOLOv8n/s | YOLOv8m/l fine |

Your Mac (MPS): treat like “8GB-class with different quirks” — prefer SD 1.5, Base depth, B0 seg.

---

## 9. Papers / mental models worth knowing

| Topic | Pointer |
|-------|---------|
| Latent diffusion | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM / Stable Diffusion) |
| Classifier-free guidance | Ho & Salimans — the math behind `guidance_scale` |
| ControlNet | Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* |
| Depth Anything V2 | Yang et al. |
| SegFormer | Xie et al. — efficient hierarchical transformers for semantic seg |
| OWL-ViT | Minderer et al. — open-vocabulary detection |
| SAM | Kirillov et al. — promptable segmentation |

---

## 10. What’s left in Notebook 1 vs Notebooks 2–3

### Notebook 1 — status

**Done (walkthrough path):**

- Load real images (RailSem19 curated samples; Nordland HF / others available)
- Depth estimation + viz
- Segmentation / RS19 GT → edit weights
- SD 1.5 + depth ControlNet snow edit + winter grade
- OWL-ViT + MobileSAM annotation on the synthetic image
- Artifact saving under `outputs/`

**Still open / deferred from this notebook:**

| Item | Notes |
|------|--------|
| **VLM judge loop** | Config stub exists (`judge.threshold: 8.5`) but no scoring cell yet |
| **True inpaint / MultiControlNet** | Intentionally avoided for SD 1.5 correctness; revisit with right checkpoints |
| **Grounded-SAM 2 swap** | Annotation quality upgrade |
| **Batch over all samples** | Notebook runs primarily on `samples[0]` for generation |
| **Realism polish** | Snow/winter still the hard part on laptop SD 1.5 — tuning + optional model upgrade |
| **More edge-case prompts** | Ice, fallen tree, fog — same pipeline, new prompts/masks |

### Notebook 2 — planned: batch dataset generation

From the notebook “What’s next” and `docs/project_summary.md`:

1. Wrap the single-image path in something like `pipeline.generate_dataset(...)`
2. Sweep prompts / seeds / curated frames
3. Persist COCO-style labels (`output.format: coco` already in root config)
4. **VLM-as-judge** (Qwen2.5-VL-7B or InternVL3-8B): prompt faithfulness, physical plausibility, annotation sanity → accept / retry / reject using `judge.threshold`
5. Optional quality metrics from `docs/quality_measure.md` (CLIP-score, LPIPS to source, KID at small N — not FID alone)

### Notebook 3 — planned: train & evaluate utility

1. Fine-tune a small detector (**YOLOv8-n** or **RT-DETR-S**) on **real only** vs **real + accepted synthetic**
2. Report **tail-class AP** (snow / obstruction / ice — the long-tail classes)
3. Image-level scores (CLIP, optional human spot-check N≈100)
4. Answer the business question: *did synthetic edge cases move the metric that matters?*

---

## 11. Practical “don’t use this blindly” checklist

1. **Depth / seg maps go to ControlNet (or masks) — not to the VLM judge.** The judge sees RGB (+ maybe the prompt and labels).
2. **Matching families matters:** SD 1.5 UNet ↔ SD 1.5 ControlNet; SDXL ↔ SDXL ControlNet; inpaint UNet ↔ inpaint pipeline.
3. **ADE20K colors** matter for seg ControlNet; GT RailSem19 ids matter for our snow weights.
4. **`strength` vs `controlnet_scale`:** first locks how much the photo can change; second locks how much depth can veto the prompt.
5. **Laptop SD 1.5 will not match FLUX photorealism.** Use it to learn the *pipeline*; upgrade the base when compute allows.
6. **Annotation models don’t verify physics.** That’s why the VLM loop is in Notebook 2.

When in doubt, change **one** knob at a time (`strength`, then `controlnet_scale`, then prompt) and keep `seed` fixed so you can see what actually moved.
