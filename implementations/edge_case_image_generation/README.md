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

We do **not** download the full ~29 GB HF zip. Toy images land in `data/mapillary_vistas/samples/`:

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

**L4:** FLUX.2-klein-4B for inpaint + instruct (default generator); SD 1.5 ControlNet depth+seg.  
**Judge:** API vision chat by default (`configs/default/judge.yaml`); optional local Qwen for offline.

## Setup

Dependencies live in the **monorepo root** `pyproject.toml` under the `edge-case-image-generation` group.

```bash
# From the repository root (EdgeCaseSynthesis/)
uv sync --dev --group edge-case-image-generation

# API key for the Vector proxy (copy template, then edit):
cp implementations/edge_case_image_generation/.env.example \
   implementations/edge_case_image_generation/.env
# paste your vp_… key into OPENAI_API_KEY in .env
# optional: HF_TOKEN=… for gated Mapillary + some model weights
```

Or run `implementations/edge_case_image_generation/scripts/setup_notebook_env.sh` to create `.venv` at the repo root and register the Jupyter kernel.

**Workshop images** (Mapillary toy subset — not the full ~29 GB zip):

- Prefer [Notebook 0](notebooks/00_flight_precheck.ipynb) (progress bar + interactive HF login), or
- CLI: `uv run python implementations/edge_case_image_generation/scripts/extract_mapillary_toy.py`

Notebooks load `.env` automatically on startup — no `export` in the terminal needed.

Notebooks add `src/` to `sys.path` automatically — no separate package install needed.

Package layout under `src/edgecase_synthesis/`:

```text
config/   data/   generate/   judge/   batch/   train/   viz/
preflight.py   bootstrap.py
```

Flat import paths (`edgecase_synthesis.pipeline`, `.batch_runner`, …) still work via thin re-exports.

## Notebooks

- `notebooks/00_flight_precheck.ipynb` — env / API / GPU / **data download** / model cache check
- `notebooks/01.5_method_comparison.ipynb` — educational bake-off of edit methods; pick `instruct` for NB1
- `notebooks/01_sample_data_generation.ipynb` — synthesis loop + fidelity / novelty / usability (VLM + gates)
- `notebooks/02_batch_dataset_generation.ipynb` — stratified split, batch synth + judge, export for NB3
- `notebooks/03_training_and_evaluation.ipynb` — YOLOv8n real-only vs real+synth on held-out real test

Supporting docs: `docs/citations.md`, `docs/figures.md`, `docs/anomaly_authoring.md`.
