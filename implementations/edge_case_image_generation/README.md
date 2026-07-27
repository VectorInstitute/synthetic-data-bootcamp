# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Pipeline

`real image → depth → segmentation → anomaly edit → open-vocab annotation → VLM judge`

Model IDs, datasets, and hardware choices live in **Hydra YAML** under `configs/`. The Python package stays generic.

## Default dataset: NEU-DET

Open steel-surface defect detection set (bbox labels, 6 classes). Good for Notebook 3: train a detector on **real vs real+accepted-synthetic** and report per-class AP.

Cite: Song & Yan, NEU Surface Defect Database. Config: `configs/data/source/neu_det.yaml` (HF id overridable).

Other sources: `local`, `urls`, `nordland_hf` (images only, no boxes).

## Hardware

```python
load_config(overrides=["hardware=cpu"])       # SD 1.5 + depth ControlNet, Qwen2.5-VL-3B
load_config(overrides=["hardware=gpu_l4"])    # SDXL inpaint, Qwen2.5-VL-7B
```

| Profile | Edit method | Judge |
|---------|-------------|-------|
| `cpu` | depth ControlNet | Qwen2.5-VL-3B (CLIP fallback) |
| `gpu_l4` | SDXL inpaint | Qwen2.5-VL-7B |

## Setup

```bash
cd implementations/edge_case_image_generation
uv sync
```

## Notebook

`notebooks/01_sample_data_generation.ipynb` — single-image walkthrough through the VLM judge.
