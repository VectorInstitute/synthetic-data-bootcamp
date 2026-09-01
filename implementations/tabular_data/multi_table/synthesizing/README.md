# ClavaDDPM synthesizer hyperparameters

Set these values in [`config.yaml`](config.yaml). The notebook pases them into `clava_synthesizing`.

ClavaDDPM does **not** take a per-table `n_samples`. The root table is sized from the real data and `sample_scale`. Child tables are then generated from the synthetic parents, using the learned distribution of how many children each parent typically has.

---

## `sampling_config`

Controls how many rows are generated and how the diffusion sampler runs. These settings apply while each parent–child pair is sampled.

### `sample_scale`

Ratio of synthesized rows to original rows for the **root** table (the table with no parent; for Berka this is `district`). Default in the toolkit is `1.0` if you omit it.

- `1.0` — about as many root rows as in the real table.
- `0.25` — about one quarter of the real root table (for example 77 districts → 19).
- `2.0` — about twice as many root rows.

Child tables are **not** sized with a separate knob. For each synthetic parent row, the synthesizer draws a group size from `all_group_lengths_prob_dicts` (saved during clustering: “given this parent cluster, how many children are typical?”). The child count is then roughly:

**number of synthetic parents × typical children-per-parent**

So shrinking the root table (smaller `sample_scale`) shrinks descendants as well, while keeping similar relationship cardinalities. The log line `Sample size: …` is `int(sample_scale * len(real_table))` for every table; for children that is a **target-scale hint**, not a hard cap. The actual child count is the sum of sampled group sizes.

This argument is passed to `clava_synthesizing(..., sample_scale=...)`. It is not a field of `ClavaDDPMSamplingConfig`.

**Trade-off:** lower values are faster and use less memory (important for large children such as `trans`). Higher values cost more compute and may be closer to the original table size for evaluation.

### `batch_size`

How many rows the diffusion model draws **per sampling step**. It does not set the total number of rows.

Larger batches usually run faster on GPU but need more memory. If sampling crashes (out of memory), lower this first. A small value such as `10` is conservative and is fine for a first run.

Used for both the unconditional root-table sampler and the classifier-guided child sampler.

### `classifier_scale`

Guidance strength when generating a **child** table so that rows match the parent’s **cluster label**.

At each denoising step a trained classifier estimates $p(y \mid x_t)$ for the parent cluster $y$. The gradient of that log-probability is multiplied by `classifier_scale` and used to steer the sample (classifier guidance, same idea as guided diffusion).

| Value | Effect |
|-------|--------|
| `0` | No guidance; child rows are not pulled toward the parent cluster. |
| `1.0` | Standard guidance (typical default). |
| `> 1` (for example `2`–`10`) | Stronger parent–cluster consistency; diversity and sample quality can drop. |
| Very large | Over-guidance: rows can collapse or look unnatural. |

The **root** table is sampled unconditionally, so this parameter does not change `district`. It only affects child generation (`conditional_sample_from_diffusion`).

---

## `matching_config`

After each parent–child pair is generated, some children have **more than one parent** (in Berka, `disp` links both `client` and `account`). Matching aligns those separately generated copies of the child so foreign keys are consistent. Tables with a single parent skip this step.

### `num_matching_clusters`

Number of FAISS IVF clusters used when searching nearest-neighbor matches between two synthetic copies of the same child (different parent keys).

- `1` — simplest index; enough for small tables.
- Larger values can speed up search on bigger tables, but the index needs enough rows to train.

This is the `n_clusters` argument to `match_tables`, not the ClavaDDPM training clusters.

### `matching_batch_size`

How many child rows are matched per FAISS search batch.

When `unique_matching` is `true`, the implementation forces this to `1` (each row is matched and then removed from the index). The YAML value then has no effect. When `unique_matching` is `false`, a larger batch is faster.

### `unique_matching`

If `true`, each row in the non-anchor child copy is used at most once (the matched index is removed from FAISS). That avoids assigning the same synthetic row to two different parent ids.

If `false`, nearest-neighbor search can reuse rows; a later pass tries to uniquify indices. Matching is faster but less strict.

### `no_matching`

If `true`, skip nearest-neighbor matching and **randomly shuffle** which parent ids are attached. Use this only as a baseline (broken relational structure). Keep `false` for a normal run.

---

## How a generation run uses these settings

1. **Root table** — sample `int(sample_scale × n_real_root)` rows, in batches of `batch_size`.
2. **Each child** — for every synthetic parent, draw a group size from the clustering checkpoint, then sample that many child rows with classifier guidance (`classifier_scale`), again in batches of `batch_size`.
3. **Multi-parent children** — match the copies with `matching_config` so both parent foreign keys sit on the same rows.

Outputs are written under `workspace_dir / exp_name / {table} / {sample_prefix}_final/` (see `GeneralConfig` in the notebook).
