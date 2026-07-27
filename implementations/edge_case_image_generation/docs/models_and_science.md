# Models and config for Notebook 1 (simplified)

Pipeline:

```text
real image
  → depth (config model_id)
  → segmentation (config model_id)
  → anomaly edit (controlnet | inpaint from hardware profile)
  → open-vocab annotate
  → VLM judge (RGB + text only)
```

**Domain knobs live in YAML**, not Python:

| Concern | Where |
|---------|--------|
| Dataset | `configs/data/source/*.yaml` |
| CPU vs L4 models | `configs/hardware/*.yaml` |
| Anomaly prompts / masks | `configs/generation/anomalies/<dataset>/` |
| Judge threshold / model | `configs/judge/` + hardware overrides |

Default dataset: **NEU-DET** (open, bbox-labeled) for a clean Notebook 3 detector demo.

Edit methods:

- `cpu` → SD 1.5 + depth ControlNet
- `gpu_l4` → SDXL inpaint

Placement masks are generic (`ellipse` / `rect` / `seg_intersection`) from anomaly YAML fractions — no cab-view geometry in code.
