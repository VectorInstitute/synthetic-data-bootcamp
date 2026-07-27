# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Pipeline

`real image → depth → segmentation → anomaly edit → open-vocab annotation → VLM judge`

Model IDs, datasets, and hardware choices live in **Hydra YAML** under `configs/`.

## Default dataset: Mapillary Vistas v2 (toy subset)

Open **street-level** scenes ([Mapillary Vistas](https://www.mapillary.com/dataset/vistas), CC BY-NC-SA) with a long-tail label set (124 classes in v2).

Workshop rare classes (synthetic targets):

- **pothole**
- **traffic_cone**
- **ground_animal**

We do **not** download the full ~29 GB HF zip. A small validation toy set (~28 images + boxes) lives under `data/samples/`, built with:

```bash
# requires: huggingface-cli login  (and accept the gated dataset terms once in the browser)
uv run python scripts/extract_mapillary_toy.py
```

Source mirror: [candylion/mapillary-vistas-v2](https://huggingface.co/datasets/candylion/mapillary-vistas-v2).  
Config: `configs/data/source/mapillary_vistas.yaml`.

Other sources: `rdd2022`, `local`, `urls`, `nordland_hf`.

## Hardware

```python
load_config(overrides=["hardware=cpu"])       # SD 1.5 inpaint + SegFormer-B2
load_config(overrides=["hardware=gpu_l4"])    # SDXL inpaint + Mask2Former + Qwen2.5-VL-7B
```

## Setup

```bash
cd implementations/edge_case_image_generation
uv sync
huggingface-cli login   # once
uv run python scripts/extract_mapillary_toy.py   # if data/samples/ is empty
```

## Notebook

- `notebooks/01.5_method_comparison.ipynb` — compare inpaint / ControlNet / instruct; pick a method per anomaly
- `notebooks/01_sample_data_generation.ipynb` — thin pipeline: load → edit (chosen methods) → annotate → judge → retry

- `notebooks/01.5_method_comparison.ipynb` — **compare inpaint vs dual ControlNet vs InstructPix2Pix** on 3 images (set `HARDWARE = "cpu"` or `"gpu_l4"`)
