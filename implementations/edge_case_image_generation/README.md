# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Pipeline

`real image → depth → segmentation → anomaly edit → open-vocab annotation → VLM judge`

Model IDs, datasets, and hardware choices live in **Hydra YAML** under `configs/`.

## Default dataset: Mapillary Vistas v2 (toy subset)

Open **street-level** scenes ([Mapillary Vistas](https://www.mapillary.com/dataset/vistas), CC BY-NC-SA) with a long-tail label set (124 classes in v2).

Workshop rare classes (synthetic targets):

- **road_debris** (cardboard box on the lane — high-contrast local insert)
- **traffic_cone**
- **fog** (global; shows why local inpaint alone is not enough)

`pothole` YAML remains available, but asphalt holes are too low-contrast for a reliable live demo.

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
load_config(overrides=["hardware=cpu"])       # SD 1.5 inpaint/CN + InstructPix2Pix + SegFormer-B2
load_config(overrides=["hardware=gpu_l4"])    # FLUX.2-klein-4B inpaint+instruct; SD 1.5 ControlNet; larger depth/judge
```

**L4 trusted local inserts:** [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) masked inpaint (`Flux2KleinInpaintPipeline`).  
**L4 instruct:** same Klein weights via `Flux2KleinPipeline`.  
**ControlNet:** still SD 1.5 depth+seg (no Klein ControlNet twin). CPU stays on the SD 1.5 / InstructPix2Pix stack.

## Setup

```bash
cd implementations/edge_case_image_generation
uv sync
huggingface-cli login   # once
uv run python scripts/extract_mapillary_toy.py   # if data/samples/ is empty
```

## Notebooks

- `notebooks/01.5_method_comparison.ipynb` — compare methods; pick `METHOD_BY_ANOMALY`
- `notebooks/01_sample_data_generation.ipynb` — thin pipeline: load → edit → annotate → judge → retry
