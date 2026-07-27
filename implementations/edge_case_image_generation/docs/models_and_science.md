# Models and config for Notebook 1

```text
real street image (Mapillary Vistas toy subset)
  → depth / seg (config model ids)
  → anomaly edit (inpaint on CPU + L4; depth ControlNet optional)
  → open-vocab annotate
  → VLM judge (RGB + text only)
```

**Default domain:** Mapillary street scenes. Workshop long-tail inserts: **pothole**, **traffic cone**, **ground animal**.

| Concern | Where |
|---------|--------|
| Dataset | `configs/data/source/mapillary_vistas.yaml` |
| CPU vs L4 models | `configs/hardware/*.yaml` |
| Anomaly prompts / masks | `configs/generation/anomalies/mapillary_vistas/` |
| Judge | `configs/judge/` + hardware overrides |

**CPU defaults:** SegFormer-B2 (seg), SD 1.5 **inpaint** (edits).  
**L4 defaults:** Mask2Former-Swin-S ADE (seg), SDXL inpaint (edits).
