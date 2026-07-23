# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for railway (and similar) perception.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Layout

```text
implementations/edge_case_image_generation/
  configs/
    hardware/       # cpu vs gpu_l4 profiles (swap with one override)
    generation/anomalies/
  src/edgecase_synthesis/
  notebooks/
  docs/
  scripts/
  data/             # Placeholders only in git — download locally
  outputs/          # Generated artifacts (gitignored)
```

## Hardware profiles

Flip the whole model stack with one Hydra override:

| Profile | Machine | Depth / Seg | Object inserts | Weather | Annotation |
|---------|---------|-------------|----------------|---------|------------|
| **`cpu`** (default) | laptop / MPS | DA-V2 Base, SegFormer-B0 | `paste` | SD 1.5 + depth CN | OWL-ViT-B + MobileSAM |
| **`gpu_l4`** | g2-standard-8 (1× L4 24 GB) | DA-V2 Large, SegFormer-B2 | **SDXL inpaint** | SDXL + depth CN | OWL-ViT-L + SAM-B |

```python
from edgecase_synthesis.config import load_config

cfg = load_config(overrides=["hardware=gpu_l4"])   # L4 VM
cfg = load_config(overrides=["hardware=cpu"])      # laptop
```

Or change the default in `configs/config.yaml`:

```yaml
defaults:
  - hardware: gpu_l4   # was: cpu
```

Anomaly YAMLs use `method: auto` so inserts follow `generation.default_anomaly_method`
from the active hardware profile (`paste` on CPU, `inpaint` on L4). Weather anomalies
keep `method: controlnet`.

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
