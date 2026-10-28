# Evals across both methods

Each script runs one method's validator on the other method's filings without changing the validator. Failures carry a `note` column that says when a failure is a difference between the two validators rather than a defect in the filing.

| Script | What it runs | On | Output |
|---|---|---|---|
| `method2_gates.py` | Method 2's Phase 7 gates (`method_2/step09_validate_filings.py`) | `filing.html` files from Method 1; the two showcase filings are in `samples/method1_showcase/` | `results/method2_gates_on_method1_showcase/`: each filing converted into Method 2's layout, plus `validation_report.csv` |
| `method1_checks.py` | Method 1's narrative checks (`method_1/validate.py`) | Method 2's 20 fictional companies in `method_2/data/synthetic/` | `results/method1_checks_on_method2_claude20/validation_report.csv` |
| `step13_gap_report.py` | Method 2's synthetic-vs-real gap report (Phase 10): length, vocabulary, numbers, retriever ranking | Method 2's synthetic and real corpora | `method_2/data/gap_report.csv` |
| `step14_privacy_check.py` | Method 2's privacy check (Phase 11): each fictional company's financials vs its real template | Method 2's fictional companies | `method_2/data/privacy_report.csv` |
| `compare_sets.py` | Fidelity, utility (retrieval) and privacy evals, scored the same way for every set | Any number of synthetic sets from either method, against one real corpus | `results/compare_sets/comparison.md`, plus per-set CSVs |

Method 2's own run of its gates on its filings is in `method_2/data/validation_report.csv`. The [folder overview](../README.md#evals) summarises all three results.

## Run

From `implementations/synthetic_10k`, in the repo's single `uv` environment (`uv sync` once from the repository root):

```bash
uv run python -m evals.method2_gates evals/samples/method1_showcase/*.html
uv run python -m evals.method1_checks
```

To check new filings from Method 1, pass their `filing.html` paths (written to `<data>/synthetic/filings/{company_id}/`) to `method2_gates.py`, with `--out-dir` set to a new folder under `results/`. `method1_checks.py` always checks every company under `method_2/data/synthetic/`.

`step13_gap_report.py` and `step14_privacy_check.py` import the `method_2` package and `../RAG/`, so run them from `implementations/synthetic_10k`:

```bash
uv run python -m evals.step13_gap_report
uv run python -m evals.step14_privacy_check
```

## Compare synthetic sets: fidelity, utility, privacy

`compare_sets.py` scores several synthetic sets side by side against the same real corpus and writes one table per question:

- **Fidelity:** section lengths, overlap of the most frequent terms, and dollar and percent density against real filings; Method 2's Phase 7 gates as `method2_gates.py` runs them, including the share of figures in Items 7 and 8 that trace to the statements.
- **Utility:** the same five financial-fact questions (revenue, net income, total assets, net and operating margin) for every set and for a sample of real filings, each answered by a verbatim span; recall@5/@10, MRR and nDCG@10 for BM25, each embedding model and a hybrid, on the set alone, the real sample alone and both mixed; and whether retrievers rank in the same order on the set as on real filings.
- **Privacy:** real S&P 500 names in the text, the template company's name or ticker (Method 2's sets), distance to the closest real company's financials, and the share of 8-word sequences that also appear in real filings, next to the same share for real filings among themselves.

The semantic retriever uses `text-embedding-3-small` through Vector's proxy (key in `OPENAI_API_KEY` or `SYNBENCH_API_KEY`), plus Method 2's local `potion-retrieval-32M` when fastembed can download it. Vectors are cached in `method_2/data/embedding_cache/`, so a rerun only embeds new text. On Coder, from `implementations/synthetic_10k`:

```bash
uv run python -m evals.compare_sets
```

By default it compares Method 1's gemini20 filings (`method_1/data/runs/gemini20/synthetic/filings`), Method 2's gemini20 batch (`method_2/data/runs/gemini20/synthetic`) and Method 2's Claude filings (`method_2/data/synthetic`), against the real filings in `method_2/data/parsed/` with XBRL figures from Method 1's `data/sec10k/financials.csv` and Method 2's `calibration.json`. `--set NAME=PATH` (repeatable), `--real-dir`, `--real-sample`, `--embedding-models` and `--out` change those. The script's docstring describes each measure.

## What each script adapts

- `method2_gates.py`: Method 1's filings show two fiscal years and don't record which real company was the style example. The script accepts prior-year values, checks all 503 S&P 500 names instead of one template, and notes year-over-year figures that Method 2's single-year check doesn't expect. Details are in its docstring.
- `method1_checks.py`: runs the checks that don't depend on Method 1's prompt format: company name, real names, number density, leaked variable names, numbers, and overlap with the template filing (only where Method 2's git-ignored `data/parsed/` exists). Method 2's single-year statements are used as both years.
