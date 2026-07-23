# Context: Reference Implementation – Synthetic Data Generation for Edge-Case AI Training

## Project Overview

This reference implementation is being developed for a company bootcamp to demonstrate an end-to-end synthetic data generation pipeline for AI applications. The objective is not to introduce a new research contribution, but to provide a clean, educational, production-inspired implementation that students can understand, modify, and extend.

The central idea is to show how a small real dataset can be expanded with high-quality synthetic edge cases using diffusion models, automatic annotation, and VLM-based quality control, and then demonstrate that these synthetic examples improve downstream model performance.

The implementation should emphasize the **pipeline** rather than any individual model.

---



# High-Level Pipeline

```
Real Images
      │
      ▼
Structure Extraction
(Depth + Segmentation)
      │
      ▼
Controlled Image Editing
(ControlNet + Diffusion)
      │
      ▼
Automatic Annotation
(Grounded SAM 2)
      │
      ▼
Quality Verification
(VLM Judge)
      │
Pass / Retry / Reject
      │
      ▼
Approved Synthetic Dataset
      │
      ▼
Fine-tune Detector
      │
      ▼
Evaluate Improvement
```

---



# Overall Goal

Given a relatively small real dataset (roughly 1,000–5,000 images):

- Generate realistic rare-event variants
- Automatically annotate them
- Automatically verify quality
- Export a clean dataset
- Demonstrate measurable downstream improvement after fine-tuning a detector

---



# Why This Pipeline Exists

Safety-critical datasets typically suffer from long-tail problems.

Example:


| Class           | Images |
| --------------- | ------ |
| Normal railway  | 99,950 |
| Ice on rail     | 15     |
| Broken rail     | 10     |
| Fallen tree     | 20     |
| Animal on track | 5      |


Collecting these rare cases is expensive.

Instead, we generate them synthetically while preserving realism.

---



# Core Insight

The project is **not** about generating entirely new images.

It is about **editing existing real images** to introduce rare conditions while preserving:

- camera viewpoint
- scene geometry
- lighting
- perspective
- deployment distribution

This produces synthetic examples that remain close to the real data distribution.

---



# Detailed Pipeline



## Step 1 — Load a Real Image

Example:

```
Sunny railway

No defects

Normal conditions
```

---



## Step 2 — Extract Structural Information

Run:

- Depth Anything V2
- SAM 2 (or Grounded-SAM)

Outputs:

### Depth Map

Represents scene geometry.

Example:

```
Close

Road

Tracks

Far

Train

Very Far

Sky
```

This is a grayscale image.

Purpose:

Preserve scene geometry during editing.

---



### Segmentation Map

Produces semantic masks.

Example:

```
Sky

Trees

Train

Rails

Road
```

Purpose:

Tell the diffusion model which pixels correspond to which objects.

---



## Important Clarification

The depth map and segmentation map are **not** inputs to the VLM.

They are conditioning inputs to **ControlNet**.

---



## Step 3 — Controlled Image Editing

Inputs:

- Original image
- Depth map
- Segmentation map
- Edit prompt

Example prompt:

```
Add heavy snow accumulation on the rails.
```

ControlNet conditions the diffusion model so that it edits the scene instead of inventing a new one.

Desired output:

```
Same camera

Same train

Same tracks

Same scene

Only snow added
```

---



## Why Use Depth and Segmentation?

Without ControlNet:

The diffusion model may change:

- camera angle
- train
- track layout
- lighting
- station

With ControlNet:

The generated image stays close to the original scene while introducing only the requested modification.

This creates controlled, realistic augmentation.

---



## Step 4 — Automatic Annotation

Run Grounded-SAM 2.

Inputs:

- synthetic image
- prompt

Example:

```
snow on rail
```

Outputs:

- bounding boxes
- segmentation masks
- labels

No manual annotation required.

---



## Step 5 — Quality Verification

Run a Vision-Language Model.

Candidate models:

- Qwen2.5-VL
- InternVL

The VLM acts as a **judge**, not an image generator.

It evaluates:

- prompt faithfulness
- physical plausibility
- annotation correctness
- edge-case presence
- image quality

Example output:

```json
{
  "prompt_faithfulness": 9,
  "physical_plausibility": 8,
  "annotation_correctness": 10,
  "edge_case_present": true,
  "overall": 9.1,
  "decision": "accept"
}
```

---



## Step 6 — Retry Loop

If the image fails:

```
Generate

↓

Judge

↓

Rejected

↓

Retry
```

Potential future improvement:

The judge explains why the image failed.

An LLM rewrites the prompt.

Generate again.

This is optional for the initial implementation.

---



## Step 7 — Approved Dataset

Each accepted image stores:

- synthetic image
- annotations
- prompt
- generation seed
- judge scores
- metadata

Example metadata:

```json
{
  "source_image": "...",
  "prompt": "...",
  "seed": 1234,
  "judge_score": 9.1,
  "accepted": true
}
```

---



## Step 8 — Fine-Tune Detector

Train:

```
YOLOv8n
```

(or similar lightweight detector)

Experiment A:

```
Real dataset only
```

Experiment B:

```
Real dataset

+

Synthetic dataset
```

Compare:

- mAP
- tail-class AP
- precision
- recall

This demonstrates the value of synthetic data.

---



# Important Discussion Outcome

The synthetic images should **not** simply duplicate every original image.

Instead:

- select different source images
- apply different edge cases
- vary severity
- vary placement

Examples:

Image 12

↓

Snow

Image 56

↓

Fallen tree

Image 233

↓

Animal crossing

Image 711

↓

Broken rail

This creates diversity without overwhelming the dataset with duplicates.

---



# Why Editing Instead of Full Generation?

Generating from scratch:

Pros:

- unlimited diversity

Cons:

- distribution drift
- unrealistic geometry
- hallucinated environments

Editing existing images:

Pros:

- realistic camera
- realistic lighting
- realistic deployment conditions
- controlled changes

This resembles counterfactual data augmentation.

---



# Notebook Structure



## Notebook 1 — Manual Walkthrough

Purpose:

Teach the pipeline.

Use approximately five images.

Show every step.

Example:

```
Load image

↓

Depth map

↓

Segmentation

↓

Generate synthetic image

↓

Annotate

↓

Judge

↓

Accept / Reject
```

Visualize every intermediate artifact.

This notebook is educational.

---



## Notebook 2 — Dataset Generation

Purpose:

Scale.

Wrap the previous steps into a reusable pipeline.

Example:

```python
pipeline.generate_dataset(...)
```

Outputs:

- synthetic images
- annotations
- metadata
- acceptance statistics

---



## Notebook 3 — Downstream Training

Purpose:

Demonstrate usefulness.

Train detector on:

- real only
- real + synthetic

Compare metrics.

This notebook provides the "business value" of the project.

---



# Recommended Repository Structure

```
project/

│
├── notebooks/
│
│   01_pipeline_walkthrough.ipynb
│
│   02_dataset_generation.ipynb
│
│   03_training_and_evaluation.ipynb
│
├── src/
│
│   generator.py
│
│   conditioning.py
│
│   annotation.py
│
│   judge.py
│
│   pipeline.py
│
│   training.py
│
│   utils.py
│
├── configs/
│
│   config.yaml
│
├── outputs/
│
│   originals/
│
│   depth/
│
│   segmentation/
│
│   synthetic/
│
│   annotations/
│
│   metadata/
│
│   rejected/
│
└── README.md
```

---



# Configuration

Prefer configuration files over hardcoded parameters.

Example:

```yaml
generator:
  model: flux-schnell

conditioning:
  use_depth: true
  use_segmentation: true

judge:
  threshold: 8.5

generation:
  retries: 3

annotation:
  prompt: "snow on rail"

output:
  format: coco
```

The notebooks should remain clean and primarily orchestrate the pipeline rather than contain implementation details.

---



# Key Architectural Clarifications

One important clarification that emerged during planning:

The diffusion model performs image generation.

The VLM does **not** generate images.

Responsibilities are separated as follows:

**Depth Anything**

- estimates scene geometry

**SAM / Grounded-SAM**

- produces semantic segmentation

**ControlNet + Diffusion**

- edits the image

**Grounded-SAM 2**

- produces annotations

**VLM**

- judges image quality

**Detector**

- measures downstream improvement

Keeping these roles distinct makes the architecture easier to understand and mirrors how modern production pipelines are typically assembled.

---



# Overall Educational Story

The reference implementation should tell a cohesive end-to-end story:

1. Start with a small real-world dataset.
2. Preserve scene structure using depth and segmentation.
3. Generate realistic edge-case variants through controlled image editing.
4. Automatically annotate the generated images.
5. Automatically verify their quality with a VLM acting as a judge.
6. Build a curated synthetic dataset.
7. Fine-tune a detector using both real and synthetic data.
8. Show measurable improvement on rare classes.

By the end of the bootcamp, participants should understand not only *how* to assemble a modern synthetic data pipeline, but *why* each component exists and how they work together to create trustworthy synthetic data that improves downstream AI systems.