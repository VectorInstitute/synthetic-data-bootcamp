# Models and config for Notebook 1

```text
real road image (RDD2022)
  → depth / seg (config model ids)
  → anomaly edit (controlnet on CPU, inpaint on L4)
  → open-vocab annotate
  → VLM judge (RGB + text only)
```

**Default domain:** open road damage (RDD2022). Potholes and foreign objects (e.g. traffic cones) are the intuitive long-tail cases.

| Concern | Where |
|---------|--------|
| Dataset | `configs/data/source/rdd2022.yaml` |
| CPU vs L4 models | `configs/hardware/*.yaml` |
| Anomaly prompts / masks | `configs/generation/anomalies/rdd2022/` |
| Judge | `configs/judge/` + hardware overrides |
