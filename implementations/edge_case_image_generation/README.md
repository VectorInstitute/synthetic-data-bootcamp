# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Pipeline

`real image → depth → segmentation → anomaly edit → open-vocab annotation → VLM judge`

Configuration is **dataset-first** under `configs/datasets/<name>/`, with shared defaults in `configs/default/` and hardware profiles in `configs/hardware/`.

## Default dataset: Mapillary Vistas v2 (toy subset)

Open **street-level** scenes ([Mapillary Vistas](https://www.mapillary.com/dataset/vistas), CC BY-NC-SA).

Workshop anomalies (see `configs/datasets/mapillary_vistas/dataset.yaml`):

- **road_debris** (cardboard box on the lane)
- **traffic_cone**
- **fog** (global; shows why local inpaint alone is not enough)

We do **not** download the full ~29 GB HF zip. Toy images land in `data/mapillary_vistas/samples/`:

```bash
# requires: huggingface-cli login  (and accept the gated dataset terms once in the browser)
uv run python scripts/extract_mapillary_toy.py
```

Other dataset packages: `rdd2022`, `nordland_hf`. Copy `configs/datasets/_template/` to add your own.

## Hardware + dataset selection

In `configs/config.yaml` (or notebook overrides):

```python
load_config(overrides=["dataset_name=mapillary_vistas", "hardware=cpu"])
load_config(overrides=["dataset_name=mapillary_vistas", "hardware=gpu_l4"])
```

**L4:** FLUX.2-klein-4B for inpaint + instruct; SD 1.5 ControlNet depth+seg.  
**CPU:** SD 1.5 inpaint / InstructPix2Pix (shared `configs/default/generation.yaml`).

## Setup

```bash
cd implementations/edge_case_image_generation
uv sync
huggingface-cli login   # once
uv run python scripts/extract_mapillary_toy.py
```

## Notebooks

- `notebooks/01.5_method_comparison.ipynb` — compare methods; pick `METHOD_BY_ANOMALY`
- `notebooks/01_sample_data_generation.ipynb` — thin pipeline: load → edit → annotate → judge → retry
