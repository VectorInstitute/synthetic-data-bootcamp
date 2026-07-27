# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Pipeline

`real image → depth → segmentation → anomaly edit → open-vocab annotation → VLM judge`

Model IDs, datasets, and hardware choices live in **Hydra YAML** under `configs/`.

## Default dataset: RDD2022 (road damage)

Open **street-level road photos** (CC BY-SA 4.0) with bbox labels for cracks and **potholes**.

Why this one:

- Everyone can read a road scene and spot a pothole / cone (unlike industrial steel textures)
- Damage is sparse — many frames are mostly clean asphalt; **potholes (D40) are the rare / severe class**
- Labeled for Notebook 3: train a detector on real vs real+accepted-synthetic, report **pothole AP**

We download the compact **China_MotorBike** subset (~183 MB). Config: `configs/data/source/rdd2022.yaml`.

Cite: Arya et al., RDD2022 / CRDDC'2022 ([sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)).

Other sources: `local`, `urls`, `nordland_hf` (images only).

Workshop anomalies (YAML): `pothole`, `alligator_crack`, `traffic_cone`.

## Hardware

```python
load_config(overrides=["hardware=cpu"])       # SD 1.5 + depth ControlNet
load_config(overrides=["hardware=gpu_l4"])    # SDXL inpaint + Qwen2.5-VL-7B
```

## Setup

```bash
cd implementations/edge_case_image_generation
uv sync
```

First notebook run downloads the RDD zip into `data/rdd2022/` and caches workshop frames under `data/samples/`.

## Notebook

`notebooks/01_sample_data_generation.ipynb` — single-image walkthrough through the VLM judge.
