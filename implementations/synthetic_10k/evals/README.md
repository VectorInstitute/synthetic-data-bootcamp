# Evals across both pipelines

Each script runs one pipeline's validator on the other pipeline's filings without changing the validator. Failures carry a `note` column that says when a failure is a difference between the two validators rather than a defect in the filing.

| Script | What it runs | On | Output |
|---|---|---|---|
| `phase7_gates.py` | Dylan's Phase 7 gates (`dylan_pipeline/src/step09_validate_filings.py`) | `filing.html` files from Amin's pipeline; the two showcase filings are in `samples/amin_showcase/` | `results/phase7_on_amin_showcase/`: each filing converted into Dylan's layout, plus `validation_report.csv` |
| `amin_checks.py` | Amin's narrative checks (`amin_pipeline/sec10k/validate.py`) | Dylan's 20 fictional companies in `dylan_pipeline/data/synthetic/` | `results/amin_checks_on_dylan20/validation_report.csv` |
| `step13_gap_report.py` | Dylan's synthetic-vs-real gap report (Phase 10): length, vocabulary, numbers, retriever ranking | Dylan's synthetic and real corpora | `dylan_pipeline/data/gap_report.csv` |
| `step14_privacy_check.py` | Dylan's privacy check (Phase 11): each fictional company's financials vs its real template | Dylan's fictional companies | `dylan_pipeline/data/privacy_report.csv` |
| `compare_sets.py` | Fidelity, utility (retrieval) and privacy evals, scored the same way for every set | Any number of synthetic sets from either pipeline, against one real corpus | `results/compare_sets/comparison.md`, plus per-set CSVs |

Dylan's own run of his gates on his filings is in `dylan_pipeline/data/validation_report.csv`. The [folder overview](../README.md#evals) summarises all three results.

## Run

From this folder, in an environment with Dylan's requirements (`../dylan_pipeline/requirements.txt`) and the repo's `uv` environment:

```bash
python phase7_gates.py samples/amin_showcase/*.html
python amin_checks.py
```

To check new filings from Amin's pipeline, pass their `filing.html` paths (written to `data/sec10k/synthetic/filings/{company_id}/`) to `phase7_gates.py`, with `--out-dir` set to a new folder under `results/`. `amin_checks.py` always checks every company under `dylan_pipeline/data/synthetic/`.

`step13_gap_report.py` and `step14_privacy_check.py` import Dylan's `src` package and `../RAG/`, so run them from `implementations/synthetic_10k`:

```bash
PYTHONPATH=dylan_pipeline:. python evals/step13_gap_report.py
PYTHONPATH=dylan_pipeline:. python evals/step14_privacy_check.py
```

## Compare synthetic sets: fidelity, utility, privacy

`compare_sets.py` scores several synthetic sets side by side against the same real corpus and writes one table per question:

- **Fidelity:** section lengths, overlap of the most frequent terms, and dollar and percent density against real filings; Dylan's Phase 7 gates as `phase7_gates.py` runs them, including the share of figures in Items 7 and 8 that trace to the statements.
- **Utility:** the same five financial-fact questions (revenue, net income, total assets, net and operating margin) for every set and for a sample of real filings, each answered by a verbatim span; recall@5/@10, MRR and nDCG@10 for BM25, each embedding model and a hybrid, on the set alone, the real sample alone and both mixed; and whether retrievers rank in the same order on the set as on real filings.
- **Privacy:** real S&P 500 names in the text, the template company's name or ticker (Dylan's sets), distance to the closest real company's financials, and the share of 8-word sequences that also appear in real filings, next to the same share for real filings among themselves.

The semantic retriever uses `text-embedding-3-small` through Vector's proxy (key in `OPENAI_API_KEY` or `SYNBENCH_API_KEY`), plus Dylan's local `potion-retrieval-32M` when fastembed can download it. Vectors are cached in `dylan_pipeline/data/embedding_cache/`, so a rerun only embeds new text. On Coder, from `implementations/synthetic_10k`:

```bash
PYTHONPATH=dylan_pipeline:. dylan_pipeline/.venv/bin/python evals/compare_sets.py
```

By default it compares Amin's gemini20 filings (`amin_pipeline/data/runs/gemini20/synthetic/filings`), Dylan's gemini20 batch (`dylan_pipeline/data/runs/gemini20/synthetic`) and Dylan's Claude filings (`dylan_pipeline/data/synthetic`), against the real filings in `dylan_pipeline/data/parsed/` with XBRL figures from Amin's `data/sec10k/financials.csv` and Dylan's `calibration.json`. `--set NAME=PATH` (repeatable), `--real-dir`, `--real-sample`, `--embedding-models` and `--out` change those. The script's docstring describes each measure.

## What each script adapts

- `phase7_gates.py`: Amin's filings show two fiscal years and don't record which real company was the style example. The script accepts prior-year values, checks all 503 S&P 500 names instead of one template, and notes year-over-year figures that Dylan's single-year check doesn't expect. Details are in its docstring.
- `amin_checks.py`: runs the checks that don't depend on Amin's prompt format: company name, real names, number density, leaked variable names, numbers, and overlap with the template filing (only where Dylan's git-ignored `data/parsed/` exists). Dylan's single-year statements are used as both years.
