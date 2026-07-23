# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for railway (and similar) perception.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Layout

```text
implementations/edge_case_image_generation/
  configs/
    hardware/       # cpu vs gpu_l4 profiles (swap with one override)
    judge/          # VLM accept / retry / reject thresholds
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

| Profile | Machine | Inserts | Judge |
|---------|---------|---------|-------|
| **`cpu`** (default) | laptop / MPS | `paste` + SD 1.5 weather | Qwen2.5-VL-**3B** (CLIP fallback) |
| **`gpu_l4`** | g2-standard-8 (1× L4 24 GB) | **SDXL inpaint** + SDXL depth CN | Qwen2.5-VL-**7B** |

```python
from edgecase_synthesis.config import load_config

cfg = load_config(overrides=["hardware=gpu_l4"])   # L4 VM
cfg = load_config(overrides=["hardware=cpu"])      # laptop
# Light laptop gate without downloading a VLM:
cfg = load_config(overrides=["hardware=cpu", "judge.backend=clip"])
```

On the L4, **unload SDXL / OWL-ViT before loading the 7B judge** (Notebook §7 does this).

## Setup

From this directory:

```bash
uv sync
# or
./scripts/setup_notebook_env.sh
```

Use a public source if you lack RailSem19 access, e.g. `data/source@data=nordland_hf`. Dataset payloads are gitignored.

## Notebooks

1. `notebooks/01_sample_data_generation.ipynb` — single-image walkthrough  
   (depth → seg → anomaly edit → annotation → **VLM judge**)
