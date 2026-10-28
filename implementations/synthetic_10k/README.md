# Synthetic 10-K reports (team BMO-1)

Two methods that build fictional companies' 10-K filings, and shared evals that check each method's output with both methods' validators. The methods were built in parallel, use different data and approaches, and are kept separate so either can be run, compared, or reused. Method 1 was built by Amin Kharaghani and Method 2 by Dylan Nguyen.

```
synthetic_10k/
  README.md       this overview
  method_1/       Method 1: Python package (python -m method_1.<step>), tests, README
  method_2/       Method 2: Python package (python -m method_2.<step>), config, data/, PLAN.md
  RAG/            retrieval benchmark and chatbot over the synthetic + real corpus (built on Method 2)
  evals/          cross-method validation scripts, the gap and privacy reports, results
  dashboard/      self-contained results dashboard and the chatbot UI (built on Method 2)
  run_batch.sh    one batch from both methods with the same model and sections, plus evals
```

## Setup

The whole repository has one environment, the `.venv` at its root. The packages both methods, RAG and the evals need are the `synthetic-10k` group in the root `pyproject.toml`, which a plain `uv sync` installs. Every command runs from this folder, so `method_1`, `method_2`, `RAG`, `evals` and `dashboard` import as packages with no `PYTHONPATH`.

```bash
uv sync                                  # once, from the repository root
cd implementations/synthetic_10k
uv run pytest method_1/tests             # Method 1's unit tests
uv run python -m method_1.narrative --help
uv run python -m method_2.step08_generate_filings --help
```

## RAG, extra evals and dashboard (from Method 2)

These modules build on Method 2 and import the `method_2` package, so run them from this folder with `uv run python -m ...`. They read and write `method_2/data/`.

| Folder | Script | Output |
|---|---|---|
| `RAG/` | `chunks.py`, `retrieval.py`: paragraph chunking; BM25, a semantic embedding retriever (fastembed locally, or `text-embedding-*` models through Vector's proxy with `RAG_EMBEDDING_MODEL`), reciprocal rank fusion and ranking metrics | |
| `RAG/` | `step11_make_qa.py`: QA pairs with a gold passage each | `method_2/data/qa_pairs.csv` |
| `RAG/` | `step12_evaluate_retrievers.py`: BM25, semantic and hybrid retrieval on synthetic, real and mixed corpora | `method_2/data/retrieval_eval.csv` |
| `RAG/` | `chatbot.py`, `chatbot_server.py`: RAG chatbot demo (CLI and browser UI) | |
| `evals/` | `step13_gap_report.py`: synthetic-vs-real gap (length, vocabulary, numbers, retriever ranking) | `method_2/data/gap_report.csv` |
| `evals/` | `step14_privacy_check.py`: distance of each fictional company's financials from its real template | `method_2/data/privacy_report.csv` |
| `evals/` | `compare_sets.py`: fidelity, utility (retrieval) and privacy for several synthetic sets side by side, with `text-embedding-3-small` | `evals/results/compare_sets/comparison.md` |
| `dashboard/` | `build_dashboard.py`: embeds the five result CSVs and vendored d3 into `template.html` | `dashboard/index.html` |

```bash
cd implementations/synthetic_10k
uv run python -m method_2.step03_build_index   # index of real + synthetic filings
uv run python -m RAG.step11_make_qa
uv run python -m RAG.step12_evaluate_retrievers
uv run python -m evals.step13_gap_report
uv run python -m evals.step14_privacy_check
uv run python -m dashboard.build_dashboard    # then open dashboard/index.html
```

Details are in [method_2/README.md](method_2/README.md) and `method_2/PLAN.md` (Phases 8-11).

## Generate a batch with both methods

[`run_batch.sh`](run_batch.sh) generates N fictional companies with each method using the same model on Vector's proxy and the same sections (Items 1, 1A, 1C, 7, 8), then runs every eval on both batches. Run it from the repository root on the Coder workspace; the settings are at the top of the script.

```bash
RUN=gemini20 N=20 MODEL=gemini-3-flash-preview bash implementations/synthetic_10k/run_batch.sh
```

## How the two methods differ

| | Method 1 ([README](method_1/README.md)) | Method 2 ([README](method_2/README.md)) |
|---|---|---|
| Real filings | 305 S&P 500 10-Ks filed in 2025 (mostly FY2024), main document only, from a local download | 50 S&P 500 companies sampled at random, FY2023 to FY2025, downloaded with edgartools, including EX-13 exhibits |
| Sections parsed | Items 1, 1A, 1C, 7, 7A, 8, plus every inline XBRL fact in the filing | Items 1, 1A, 1C, 7, 8; XBRL facts from SEC's companyfacts API |
| Calibration | Per GICS sector: percentiles of about 20 ratios and of section lengths | Generic across sectors: percentiles of 6 totals and 6 ratios |
| Fictional company | Ratios sampled from its sector's percentiles | One real company-year as a template, scaled by a random size factor |
| Statements | Two years (current and prior), income statement, balance sheet, cash flow | One year, line items from the template company's own ratios |
| Narrative | Written by an LLM through Vector's proxy (Gemini by default), with checks and up to 3 retries per section | Written in a Claude Code session; the API script (`step08`) is built but has not been run |
| Output | Items 1 to 16 as text, plus an EDGAR-style HTML and PDF filing | Items 1, 1A, 1C, 7, 8 as text (condensed by design) |
| Built-in checks | `validate.py`, run on each section during generation | `step09_validate_filings.py`, run on the finished filings |

## Evals

[`evals/`](evals/README.md) runs each method's validator on the other method's filings, without changing either validator.

| Validator | Run on | Checks | Pass | Fail | Result |
|---|---|---|---|---|---|
| Method 2's step09 | its own 20 companies | 360 | 260 (+100 info) | 0 | [report](method_2/data/validation_report.csv) |
| Method 2's step09 | Method 1's 2 showcase filings | 36 | 22 (+10 info) | 4 | [report](evals/results/method2_gates_on_method1_showcase/validation_report.csv) |
| Method 1's checks | Method 2's 20 companies | 400 (+100 skipped) | 375 | 25 | [report](evals/results/method1_checks_on_method2_claude20/validation_report.csv) |

None of the failures is a defect in a filing. Each one comes from a difference in what the two validators expect:

- **Method 2's checks on Method 1's filings:** Method 2's Item 8 heading pattern accepts only "Statement of Operations", not the plural that Method 1's filings use (2 failures). Its number check expects single-year filings, so it flags year-over-year changes and prior-year margins in MD&A (2 failures, 17 figures). Every figure in Item 8 traces to the statements.
- **Method 1's checks on Method 2's filings:** Method 1's number check allows fewer derived figures. Every figure behind the 25 failures is one that Method 2's check accepts, such as a sum of lines (total debt) or a ratio Method 1's check doesn't compute. No real company names, leaked variable names or missing company names were found.
- The exemplar-overlap check (copying from the real template filing) was skipped for Method 2's filings, because its parsed real filings are git-ignored. It runs automatically where `method_2/data/parsed/` exists.
