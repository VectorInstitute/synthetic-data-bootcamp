# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for railway (and similar) perception.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Layout

```text
implementations/edge_case_image_generation/
  configs/          # Hydra configs (data, conditioning, generation anomalies, …)
  src/edgecase_synthesis/
  notebooks/        # Notebook 1 walkthrough (+ later notebooks)
  docs/             # Design notes, model guide
  scripts/          # Env setup helpers
  data/             # Placeholders only in git — download locally
  outputs/          # Generated artifacts (gitignored)
```

## Setup

From this directory:

```bash
# with uv
uv sync

# or
./scripts/setup_notebook_env.sh
```

Place RailSem19 (or other) data under `data/` as described in the notebook. Dataset files are not committed.

## Notebooks

1. `notebooks/01_sample_data_generation.ipynb` — single-image pipeline walkthrough  
   (depth → seg → anomaly edit → open-vocab annotation)

## Note on generation methods

Object inserts (cone / branch / animal) use a fast `paste` method by default because
depth ControlNet resists inventing new objects on the track plane. Weather-style
edits (e.g. `snow_on_rails`) use `method: controlnet`. See anomaly YAMLs under
`configs/generation/anomalies/`.
