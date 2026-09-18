# Pipeline figures (workshop)

Schematics for Notebooks 1 and 3. Rendered as Mermaid in the notebooks; this
file is the source of truth if you export slides later.

## F1 — Synthesis loop

```text
 real photo
     │
     ▼
 ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────────────┐   ┌───────┐
 │  LOAD   │ → │  EDIT   │ → │ ANNOTATE │ → │ JUDGE (VLM + gates) │ → │ RETRY │─┐
 └─────────┘   └─────────┘   └──────────┐   └──────────┬──────────┘   └───────┘ │
                                        │              │ accept                 │
                                        │              ▼                        │
                                        │         training set                  │
                                        └───────────────────────────────────────┘
```

**Judge** = API VLM scores **and** hard gates (boxes, placement, VLM fidelity,
CLIP embedding safe-zone). Any hard fail → retry (new variation) or reject.

## F2 — Fidelity / novelty safe-zone (embedding view)

```text
                 too novel / OOD
                      ▲
                      │  ✗  fail fidelity
                      │     (far from real bank)
        ──────────────┼──────────────► neighbor similarity
                      │
              ✓ SAFE  │  ✗ fail novelty
              ZONE    │     (near-dupe of real∪accepted)
                      │
```

- **Fidelity (global/local):** cosine sim to nearest **real** same-class image
  must be ≥ `min_real_sim_*`.
- **Novelty:** cosine sim to nearest **real ∪ accepted synth** must be
  ≤ `max_neighbor_sim`.

VLM `global_fidelity` / `object_fidelity` are a parallel, semantic gauge of the
same idea (reference photos in the judge prompt).

## F3 — Downstream dose–response

Plot rare-class AP@0.5 (and mAP) for:

`real_only` → `real_synth_50` → `real_synth_100`

on a **fixed held-out real test**. More synth should not be assumed monotonic —
see Notebook 3 discussion.
