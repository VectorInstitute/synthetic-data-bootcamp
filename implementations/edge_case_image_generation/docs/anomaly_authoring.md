# Anomaly authoring cheat-sheet

How to define a new rare class for this bootcamp pipeline. Keep this page next to
`configs/datasets/_template/generation/anomalies/example_anomaly.yaml` — Notebook 1
stays focused on the science; this page is the config playbook.

## Where files live

```text
configs/datasets/<dataset>/
  dataset.yaml                 # workshop_anomalies list, source_hint
  generation/anomalies/<id>.yaml
```

Add your id to `workshop_anomalies` (and NB1 `METHOD_BY_ANOMALY` / NB2 maps).

## Keys that matter

| Key | Purpose |
|-----|---------|
| `id` / `display_name` | Stable id + human name for the judge |
| `prompt` / `prompt_template` | Base edit instruction (method-specific fields vary) |
| `variations` | Axes shuffled→cycled for novelty across retries / batch |
| `edit_mask` | Ellipse / road_patch / … for inpaint & placement priors |
| `annotation_classes` | Open-vocab YOLO-World query phrases |
| `accept_gates` | Box size / vertical band / bottom-edge rejects (hood paste) |

## Minimal workflow

1. Copy `_template/.../example_anomaly.yaml` → `anomalies/my_class.yaml`.
2. Fill prompts + `annotation_classes` that match how the object looks in photos.
3. Tune `edit_mask` on one seed in Notebook 1 (inpaint) or rely on instruct + gates.
4. Set `accept_gates` after you see failure modes (hood paste → `max_bottom_edge_frac`).
5. Add variation axes so retries are not identical pastes.
6. Smoke-test in NB1 → scale in NB2 → measure in NB3.

## Fidelity vs novelty knobs (not in the anomaly file)

- VLM fidelity: `configs/default/judge.yaml` → `fidelity_threshold`, reference counts.
- Embedding safe-zone: `judge.embedding_gate` (`min_real_sim_*`, `max_neighbor_sim`).

See [citations.md](citations.md) for the science framing behind gates.
