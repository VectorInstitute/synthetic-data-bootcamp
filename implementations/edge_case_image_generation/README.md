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

We do **not** download the full ~29 GB HF zip. Toy images land in `data/mapillary_vistas/samples/`.

**Preferred:** run [Notebook 0](notebooks/00_flight_precheck.ipynb) — it checks the cache and downloads the toy subset with a progress bar (plus interactive HF login if needed). You do **not** need a separate CLI extract for the workshop path.

Optional CLI (same extract as Notebook 0):

```bash
# requires: accept Mapillary terms on HF once
# (https://huggingface.co/datasets/candylion/mapillary-vistas-v2), then huggingface-cli login
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

Dependencies live in the **monorepo root** `pyproject.toml` under the `edge-case-image-generation` group. Library code ships with the workspace package **`aieng-synthetic-data`** as `aieng.syn_data.image` (same pattern as `aieng.syn_data.text` / `.synbench`).

```bash
# From the synthetic-data-bootcamp repository root
uv sync --dev --group edge-case-image-generation

# Optional local overrides (Vector key is usually already in the workshop env):
cp implementations/edge_case_image_generation/.env.example \
   implementations/edge_case_image_generation/.env
# edit only if you want your own API keys / base URLs, or HF_TOKEN=…
```

Or run `implementations/edge_case_image_generation/scripts/setup_notebook_env.sh` to create `.venv` at the repo root and register the Jupyter kernel.

**Workshop images** (Mapillary toy subset — not the full ~29 GB zip):

- Prefer [Notebook 0](notebooks/00_flight_precheck.ipynb) (progress bar + interactive HF login), or
- CLI: `uv run python implementations/edge_case_image_generation/scripts/extract_mapillary_toy.py`

Notebooks load `.env` automatically on startup when present — no `export` in the terminal needed. If the workshop machine already exports `OPENAI_API_KEY` / proxy URL, you can skip creating `.env`.

Imports use the installed package (no `sys.path` hacks):

```python
from aieng.syn_data.image.config import load_config
from aieng.syn_data.image.bootstrap import bootstrap_project_root
```

Package layout under `aieng-synthetic-data/aieng/syn_data/image/`:

```text
config/   data/   generate/   judge/   batch/   train/   viz/
preflight.py   bootstrap.py
```

Flat import paths (`aieng.syn_data.image.pipeline`, `.batch_runner`, …) still work via thin re-exports.

This folder keeps **configs**, **data**, **notebooks**, and **scripts** only.

## Notebooks

- `notebooks/00_flight_precheck.ipynb` — env / API / GPU / **data download** / model cache check
- `notebooks/00.5_method_comparison.ipynb` — educational bake-off of edit methods; pick `instruct` for NB1
- `notebooks/01_sample_data_generation.ipynb` — synthesis loop + fidelity / novelty / usability (VLM + gates)
- `notebooks/02_batch_dataset_generation.ipynb` — stratified split, batch synth + judge, export for NB3
- `notebooks/03_training_and_evaluation.ipynb` — YOLOv8n real-only vs real+synth on held-out real test

Supporting docs: `docs/citations.md`, `docs/figures.md`, `docs/anomaly_authoring.md`.
