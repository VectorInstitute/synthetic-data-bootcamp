# Models and config for Notebook 1

```text
real street image (Mapillary Vistas toy subset)
  → depth / seg (config model ids)
  → anomaly edit (inpaint / ControlNet / instruct)
  → open-vocab annotate
  → VLM judge (RGB + text only)
```

**Default domain:** Mapillary street scenes. Workshop inserts: **road_debris**, **traffic_cone**, **fog**.

| Concern | Where |
|---------|--------|
| Dataset package | `configs/datasets/<name>/` (`dataset.yaml`, `data.yaml`, anomalies) |
| Root knobs | `configs/config.yaml` → `dataset_name` + `hardware` |
| CPU vs L4 models | `configs/hardware/*.yaml` |
| Shared defaults | `configs/default/{paths,conditioning,generation,annotation,judge}.yaml` |
| Anomaly prompts / masks | `configs/datasets/<name>/generation/anomalies/` |
| Judge | `configs/default/judge.yaml` + hardware overrides; `source_hint` on `dataset.yaml` |
| Auto-label | YOLO-World boxes (`yolov8s-worldv2.pt`); no SAM — training target is detection |

**CPU defaults:** SegFormer (seg), SD 1.5 inpaint + InstructPix2Pix.  
**L4 defaults:** stronger seg, FLUX.2-klein-4B for inpaint + instruct; SD 1.5 ControlNet depth+seg.
