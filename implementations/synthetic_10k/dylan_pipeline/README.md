# Real 10-K Corpus (SEC EDGAR)

Downloads and parses real 10-K filings for 50 randomly sampled S&P 500 companies. This corpus
is the real-data side of a synthetic-10-K benchmark: it provides style templates, a comparison
set, and XBRL facts.

**Status:** Steps 1-5 (project setup, 10-K download, section parsing, metadata index, XBRL
companyfacts) and Phases 3-11 (calibration profile, fictional-company specs, a scenario per
company, financial statements, narrative generation, validation, same-layout index, QA and
retrieval evaluation, synthetic-vs-real gap report, privacy check) are all implemented — every
planned phase except Phase 7's human review, which needs an actual person. The evaluation is
organized around three questions (see "Evaluation framework" in `PLAN.md`): **fidelity** (does it
look and compute like a real 10-K — `data/validation_report.csv`, `data/gap_report.csv`),
**utility** (is it actually useful for building a retrieval benchmark —
`data/qa_pairs.csv`/`data/retrieval_eval.csv`, and whether retriever ranking transfers to the real
corpus), and **privacy** (no real company name/ticker leak, and no fictional company's financials
sit close enough to its real template to make the disguise pointless —
`data/privacy_report.csv`). See `PLAN.md` for detail, or `dashboard/index.html` for all of it laid
out.

## Layout

In this repository, Dylan's RAG modules (`chunks`, `retrieval`, `chatbot`, `chatbot_server`, steps 11-12) are in `../RAG/`, steps 13-14 are in `../evals/`, and step 15 plus `dashboard/` are in `../dashboard/` (as `build_dashboard.py`). They still import this folder's `src` package, so run them from `implementations/synthetic_10k` with `PYTHONPATH=dylan_pipeline`, as the commands below show. The tree below is the original single-folder layout.

```
config.yaml              SEC identity placeholders, company lists, fiscal years (committed)
config.local.yaml        your real SEC identity (git-ignored, create it yourself)
requirements.txt         edgartools, PyYAML, BeautifulSoup, lxml (pinned)
src/
  config.py                   config loading, shared paths, identity check
  utils.py                    logging setup, retry with backoff, atomic file writes
  html_to_text.py             convert 10-K HTML to readable plain text
  # Pipeline scripts, numbered in run order (see "Command sequence" in PLAN.md):
  step00_sample_companies.py  reproducibly sample config.yaml's 50-ticker company list
  step01_download_10k.py      download 10-K filings
  step02_parse_sections.py    split 10-K text into Items 1, 1A, 1C, 7, 8
  step03_build_index.py       data/index.csv, one row per parsed section, real + synthetic (Phase 8)
  step04_fetch_companyfacts.py  download XBRL companyfacts per company
  step05_make_calibration.py  data/calibration.json from companyfacts (Phase 3)
  step06_make_specs.py        fictional-company specs from calibration.json (Phase 4)
  step06b_add_scenarios.py    adds a scenario (M&A, product launch, etc.) to each spec (Phase 4.5)
  step07_make_financials.py   balance sheet, income statement, cash flow (Phase 5)
  step08_generate_filings.py  Claude API narrative generation (Phase 6, built but not run)
  step09_validate_filings.py  automatic validation gates: numeric, leaks, headings (Phase 7)
  step10_build_pdfs.py        renders each filing (real-style cover + Items 1/1A/1C/7/8) to a PDF
  chunks.py                   shared paragraph-level chunking of index.csv's item files (Phase 9)
  retrieval.py                BM25, a real semantic embedding retriever, reciprocal rank fusion, metrics (Phase 9)
  step11_make_qa.py           data/qa_pairs.csv: synthetic (deterministic) + real (hand-authored) QA (Phase 9)
  step12_evaluate_retrievers.py  data/retrieval_eval.csv: BM25/semantic/hybrid x 3 experiments (Phase 9)
  step13_gap_report.py        data/gap_report.csv: length, vocab, numbers, retriever ranking (Phase 10)
  step14_privacy_check.py     data/privacy_report.csv: distance from each company's real template (Phase 11)
  step15_build_dashboard.py   embeds the five result CSVs + vendored d3 into dashboard/template.html -> dashboard/index.html
  chatbot.py                  RAG chatbot over the synthetic corpus (ad hoc demo, CLI)
  chatbot_server.py           browser UI for chatbot.py, served alongside dashboard/ (ad hoc demo)
data/
  sp500_constituents.csv frozen S&P 500 snapshot src.step00_sample_companies samples from (committed)
  raw/{ticker}/{fy}/     10k.htm, 10k.txt, ex13_*.htm (if any), metadata.json
  parsed/{ticker}/{fy}/  item_1.txt, item_1a.txt, item_1c.txt, item_7.txt, item_8.txt
  companyfacts/{ticker}.json   raw XBRL companyfacts per company (git-ignored, ~180 MB; prebuilt
                               copy: `gh release download companyfacts-v1 -R
                               dylanwang2111/synthetic-data-bootcamp -p "sec10k_companyfacts.tar.gz"`,
                               then `tar -xzf sec10k_companyfacts.tar.gz -C data`)
  calibration.json       percentile distributions and per-company-year records (committed)
  index.csv              one row per parsed section, real + synthetic (git-ignored, reproduce when needed)
  download_status.csv    one row per company-year from the last download run
  parse_status.csv       one row per company-year-item from the last parse run
  companyfacts_status.csv one row per company from the last companyfacts fetch
  synthetic/{fictional_id}/{fy}/spec.json        fictional-company specs (committed)
  synthetic/{fictional_id}/{fy}/financials.json  its 3 financial statements (committed)
  validation_report.csv  one row per fictional company x check from the last validation run
  pdf_reports/{fictional_id}_{fy}.pdf  rendered filing, one PDF per company-year (git-ignored, derived)
  qa_pairs.csv            130 QA pairs (118 synthetic, 12 real; 115 factual, 15 analytical), one row per question (committed)
  retrieval_eval.csv      recall@5/@10, MRR, nDCG@10 per retriever x experiment (committed)
  gap_report.csv          12 rows: section length, vocabulary, number density, retriever ranking (committed)
  privacy_report.csv      112 rows: each company's financials vs. its real template's actual values (committed)
logs/                    per-script log files
dashboard/template.html  dashboard source: live-fetches the five data/*.csv above (for editing, via a server)
dashboard/vendor/        vendored d3.v7.9.0.min.js, so the built dashboard needs no CDN
dashboard/index.html    generated by step15_build_dashboard.py: template.html with the five CSVs + vendored d3 embedded
dashboard/chatbot.html  chatbot UI, served by chatbot_server.py alongside index.html (ad hoc demo)
```

## Setup

From this directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Then create `config.local.yaml` with your identity. This file is git-ignored, so your name and
email never reach the public repo:

```yaml
sec_identity:
  name: "Your Name"
  email: "you@example.com"
```

Values here override the placeholders in `config.yaml`. SEC requires a real name and contact
email on every request. The scripts exit with an error if the identity is missing or still a
placeholder.

## Download 10-K filings

```bash
python -m src.step01_download_10k
```

- `run_profile` in `config.yaml` selects the company list: `test` (JPM, PNC, FITB, KEY, ZION, a
  small sanity-check set) or `full` (50 S&P 500 companies sampled with `src.step00_sample_companies`;
  see its docstring for how and why). Set `fiscal_years` to change the years.
- Re-running skips any company-year whose folder is already complete.
- Missing company-years (no 10-K for that fiscal year, or ticker not found) are logged and
  recorded in `download_status.csv`. The run continues.
- Rate limiting: edgartools throttles requests to 9 per second by default, under SEC's
  10/second limit. Network errors and HTTP 429 responses are retried with backoff by edgartools.
  The download script adds another retry layer for other failures.

## Parse 10-K sections

```bash
python -m src.step02_parse_sections
```

- Splits each `10k.txt` into Items 1, 1A, 1C, 7, and 8 (see `SECTIONS.md` for why these five).
  The table of contents repeats every Item heading; the occurrence with the most text before
  the next heading is taken as the body.
- When an item is only "incorporated by reference" instead of written out, the heading named in
  that cross-reference sentence is located inside the filing's EX-13 annual report exhibit
  (PGR, USB, WFC in this corpus), if it has one, or elsewhere in the 10-K body otherwise.
- A heading reprinted as a running header on every page (seen in a few large banks and insurers)
  is detected by an unusually high occurrence count, so the result isn't just one page of it.
- Always reparses everything in `config.yaml`'s company list and fiscal years; it's local and
  fast, so there's no skip-if-done logic to go stale.
- Every section, found or not, gets a row in `parse_status.csv` (`ok`, `short`, or `missing`,
  plus character count and extraction method). Review `short` and `missing` rows before relying
  on a given section.

## Build the index and fetch XBRL companyfacts

```bash
python -m src.step03_build_index
python -m src.step04_fetch_companyfacts
```

- `step03_build_index.py` writes `data/index.csv`, one row per parsed section (ticker, company name,
  CIK, fiscal year, filing date, accession number, item, path, character count, word count,
  source). Local and fast; always rewrites the whole file. Also indexes
  `data/synthetic/{fictional_id}/{fy}/item_*.txt` (Phase 8) — `source` is `real` or `synthetic`,
  and CIK/filing date/accession number are blank on synthetic rows since there's no real-world
  equivalent. Re-run it after the synthetic corpus changes, not just after the real one.
- `step04_fetch_companyfacts.py` downloads each company's XBRL companyfacts JSON from
  `data.sec.gov/api/xbrl/companyfacts/`, using the CIK from its already-downloaded
  `metadata.json`. Re-running skips any ticker whose file already exists.

## Build the calibration profile

```bash
python -m src.step05_make_calibration
```

- Reads every `data/companyfacts/{ticker}.json`, pulls total assets, total liabilities,
  stockholders' equity, revenues, net income, and operating income for each fiscal year in
  `config.yaml` (trying a short list of fallback XBRL tags per metric, since the same line item
  is tagged differently across filers), adds six derived ratios, and writes `data/calibration.json`
  with the full per-company-year records plus each metric's percentile distribution.
- Calibration is generic across sectors, not bank-specific — the company list is a random S&P 500
  sample now, not a bank list. See `PLAN.md` Phase 3.
- Not every company reports every tag; `calibration.json`'s `missing` field says which
  company-year is missing which metric, rather than silently dropping or guessing it.

## Build fictional company specs

```bash
python -m src.step06_make_specs
```

- Samples 20 real company-years from `calibration.json` as templates (no two fictional companies
  share one) and writes `data/synthetic/{fictional_id}/{fy}/spec.json` for each: a fictional name,
  ticker, HQ state, sector and sub-industry (carried over from the template's real company),
  financials, and the Item 1C cybersecurity facts a filing needs.
- Financials are the template's scaled by a random size factor and rebuilt from its own ratios
  (jittered), not resampled independently, so assets = liabilities + equity holds by construction
  and is then checked; a spec that somehow failed the check would be retried (up to 5 times) with
  a new random draw. See `PLAN.md` Phase 4 for why (keeps the fictional company recognizably the
  same kind of company as its template, and keeps every table reconciling for Phase 5).
- Every fictional name and ticker is checked against all 503 real S&P 500 names and tickers, not
  just the 50 downloaded, and against every other fictional company already generated.
- Deterministic: re-running regenerates the same 20 specs (fixed seed in `src/step06_make_specs.py`).

## Add a scenario to each fictional company spec

```bash
python -m src.step06b_add_scenarios
```

- Adds a `scenario` field to each `spec.json` (M&A, a product launch, a regulatory event, a
  margin crisis, and so on, or `baseline` for ordinary-course companies) with structured facts the
  narrative must ground its claims in. See `PLAN.md`'s Phase 4.5 for why: without this, the
  narrative had nothing distinguishing to say beyond generic sector-template boilerplate, and there
  was no structured ground truth for qualitative claims the way there is for numbers.
- Every dollar figure a scenario cites is copied from that company's own, already-reconciled
  `financials.json` (see the script's `SCENARIOS` dict for exactly which field each one came from),
  so this never touches `financials.json` and Phase 7's numeric gate is unaffected.
- Scenario assignments are hardcoded per company (editorial content, like Phase 6's narrative),
  not randomly generated.

## Build fictional company financial statements

```bash
python -m src.step07_make_financials
```

- Writes `data/synthetic/{fictional_id}/{fy}/financials.json` for every spec: a balance sheet,
  income statement, and cash-flow statement, each reconciling exactly to `spec.json`'s totals.
- Every line item is the template company's own ratio (that line divided by its own total, pulled
  from `data/companyfacts/{ticker}.json`) applied to the fictional company's total — the same
  "scale the template's own shape" approach `step06_make_specs.py` uses, one level more detailed.
  Whatever a ratio basis can't explain (a tag the template doesn't report, or simply what's left)
  becomes one named "other" line per statement/section, which is also what forces each one to
  reconcile exactly rather than approximately.
- Income tax expense is always the residual between pretax income and net income (never ratio-
  based), so net income ties out exactly whether or not the template reported an operating income
  figure to tie to as well.
- Deterministic and local: no network calls, reproducible from `spec.json` and the already-cached
  companyfacts, fixed seed in `src/step07_make_financials.py`.

## Run a separate batch

Steps 06 to 09 read and write `data/synthetic/` unless `SYNTHETIC_DIR` is set, so a new
batch can sit next to the committed 20 companies:

```bash
export SYNTHETIC_DIR=data/runs/gemini20/synthetic
python -m src.step06_make_specs --n 20 --seed 7
python -m src.step07_make_financials
python -m src.step08_generate_filings --backend openai --model gemini-3-flash-preview --workers 4
python -m src.step09_validate_filings   # report: data/runs/gemini20/validation_report.csv
```

`--backend openai` calls any OpenAI-compatible endpoint, such as Vector's proxy on the
Coder workspace (`SYNBENCH_BASE_URL`, key in `OPENAI_API_KEY` or `SYNBENCH_API_KEY`).
Step 07 needs `data/companyfacts/` (step 04) and step 08 uses `data/parsed/` (steps
01-02) for style exemplars; both are git-ignored.

## Generate fictional company narrative text (placeholder, not run)

```bash
python -m src.step08_generate_filings
```

- **Not currently used.** The `item_*.txt` files under `data/synthetic/` were written directly in
  a Claude Code session instead, so this script has never been run against the real API and has
  cost nothing. See `PLAN.md` Phase 6 and open question 3.
- Requires the `anthropic` package (in `requirements.txt`) and `ANTHROPIC_API_KEY` set in the
  environment; exits immediately with an explanation if the key isn't set.
- Per item, builds a prompt from the spec's facts, the relevant computed figures from
  `financials.json`, a real exemplar from the template company's own `data/parsed/` (style and
  structure only, not to be copied), a length target (the real corpus's median for that item, from
  `parse_status.csv`), and whatever's already been generated for that company, then calls the
  Claude API and writes the result.

## Validate fictional company narrative text

```bash
python -m src.step09_validate_filings
```

- Writes `data/validation_report.csv` (committed): one row per fictional company per check. Current
  result: 0 failures across 373 checks on all 20 companies.
- Checks every dollar figure and percentage quoted in Item 7 and Item 8 against `financials.json`
  (a leaf value, a handful of named combinations such as "total debt," or a standard ratio like net
  margin), that no real template company's name or ticker appears, that Item 8 has all three
  statement headings, that every item file is substantive, and that every non-baseline company's
  `spec.json` scenario (Phase 4.5) is actually grounded somewhere in its narrative. Section length
  against the real corpus's distribution is reported but not gated, since Phase 6 condensed every
  section by a deliberate choice (see `PLAN.md` Phase 6).
- Covers Phase 7's two automatic/model-judged gates; human review of about 10 filings is the one
  gate this script can't do. See `PLAN.md` Phase 7.

## Add the synthetic corpus to the index (Phase 8)

```bash
python -m src.step03_build_index
```

- Re-running `step03_build_index.py` after the synthetic corpus exists appends its rows to the
  same `data/index.csv` used for the real corpus, each tagged `source = synthetic`; `cik`,
  `filing_date`, and `accession_number` are blank on those rows. Current result: 829 rows (729
  real, 100 synthetic — 20 companies x 5 items). Retrieval code (Phase 9) filters on `source`
  rather than needing a second index. See `PLAN.md` Phase 8.

## Build QA pairs and evaluate retrieval (Phase 9)

```bash
PYTHONPATH=dylan_pipeline python -m RAG.step11_make_qa
PYTHONPATH=dylan_pipeline python -m RAG.step12_evaluate_retrievers
```

- `step11_make_qa.py` writes `data/qa_pairs.csv` (committed): 118 synthetic questions, generated
  deterministically from each company's own `financials.json`/scenario facts with no model call,
  and 12 real questions hand-authored from a 6-company sample (AEP, BEN, USB, CVS, NOC, CRWD — the
  same sample Phase 7's model-judged review used). Every answer is verified as a verbatim substring
  of its gold passage before being written; the script hard-fails if one isn't found. Every
  question also carries a `style` — `factual` (115 rows: a figure or named entity) or `analytical`
  (15 rows: why something happened, or what it implies — still one short exact answer, but finding
  it means reading a passage that connects a cause to an effect). See `PLAN.md` Phase 9 for the
  three ways analytical questions are grounded.
- `step12_evaluate_retrievers.py` writes `data/retrieval_eval.csv` (committed): recall@5, recall@10,
  MRR, and nDCG@10 for BM25, a real semantic embedding retriever (`fastembed` +
  `minishlab/potion-retrieval-32M` — see `PLAN.md` Phase 9 for why that model specifically), and
  their reciprocal-rank-fusion hybrid, across three experiments (`synthetic_only`, `real_only`,
  `mixed`).
- See `PLAN.md` Phase 9 for the full results table and what they mean (short version: BM25 edges
  out the semantic retriever on all three experiments, but the gap is much smaller on the real
  sample than an earlier TF-IDF version showed).

## Quantify the synthetic-vs-real gap (Phase 10)

```bash
PYTHONPATH=dylan_pipeline:. python evals/step13_gap_report.py
```

- Writes `data/gap_report.csv` (committed): section length, vocabulary overlap, number-figure
  density, BM25 score distribution, and whether the best-performing retriever agrees between the
  `synthetic_only` and `real_only` experiments in `data/retrieval_eval.csv`. Scans the *full* real
  corpus for the length/vocabulary/number comparisons (not just the 6-company sample used
  elsewhere) since none of those need the dense-vector corpus's matrix.
- Short version (see `PLAN.md` Phase 10 for the full table and caveats): synthetic sections run
  3.6%-31.2% of real's median length per item; synthetic narrative states 2-3x more numbers per
  1,000 words than real's more qualitative MD&A prose; the full three-way retriever ranking
  (`bm25`, `hybrid_rrf`, `semantic`) agrees between real and synthetic.

## Check privacy: distance from each company's real template (Phase 11)

```bash
PYTHONPATH=dylan_pipeline:. python evals/step14_privacy_check.py
```

- Writes `data/privacy_report.csv` (committed): for each fictional company and each financial
  field, the percent difference between the fictional value and its real template's actual
  same-year value (from `data/calibration.json`, written before any fictional company existed).
  Answers a question Phase 7's name/ticker leak check doesn't: even with no literal leak, are the
  fictional numbers just the real company's own numbers relabeled?
- Short version (see `PLAN.md` Phase 11 for the full per-company breakdown): median absolute
  difference per field is 20.0% across all 20 companies; no company is within 5% of its template on
  every field.

## View the results dashboard

```bash
PYTHONPATH=dylan_pipeline python dashboard/build_dashboard.py
```

Then just open `dashboard/index.html` directly — double-click it, or open it from a file manager.
It's a self-contained page (vanilla JS + d3): the build step above embeds the current contents of
`data/validation_report.csv`, `data/qa_pairs.csv`, `data/retrieval_eval.csv`, `data/gap_report.csv`,
and `data/privacy_report.csv`, plus the vendored `dashboard/vendor/d3.v7.9.0.min.js`, directly into
it — no CDN script tag, so no server and no network access of any kind is needed to view it. It's
organized around the three questions a synthetic corpus has to answer (see "Evaluation framework"
in `PLAN.md`): fidelity, utility, and privacy.

Like every other output in this pipeline, the built file goes stale the moment those five CSVs
change — re-run the build step above after re-running any of them, the same as you'd re-run
`step12` after `step11`.

`dashboard/template.html` is the editable source (markup, CSS, chart code); `dashboard/index.html`
is generated from it and shouldn't be hand-edited. To iterate on the template itself, serve the
directory instead — with no embedded data, the template falls back to a live, cache-busted fetch
of the same five files:

```bash
cd implementations/synthetic_10k && python -m http.server
```

then open `http://localhost:8000/dashboard/template.html`.

## Try the RAG chatbot demo (ad hoc)

`src/chatbot.py` is a small demo app on top of Phase 9's retrievers: BM25 + the real
semantic embedding retriever (`RAG.retrieval`, fastembed / `minishlab/potion-retrieval-32M`),
fused via reciprocal rank fusion, over the 20 synthetic companies. It has a scripted mode
that asks one analytical question per scenario company (13 of them — M&A, margin crisis,
cash burn, regulatory pressure, ...), shows that company's scenario summary from its
`spec.json`, and reports whether retrieval actually lands on the scenario-grounded passage
(checked against `data/qa_pairs.csv`'s `gold_chunk_id`) rather than just a
keyword-overlapping one.

Retrieval works with no setup. Generation is optional and picked at startup from whichever
key is set: `DEEPSEEK_API_KEY` first, then `ANTHROPIC_API_KEY`, then retrieval-only (falls
back to showing the top-retrieved passage itself) — the same "retrieval needs no key,
generation does" split as `step08_generate_filings.py`. Put a key in a gitignored `.env`
file in this directory (`DEEPSEEK_API_KEY=sk-...`), or export it in your shell; either way,
never commit it.

The semantic retriever's embedded doc matrix is cached to
`data/semantic_embeddings_cache.npz` (git-ignored, numpy's native archive — see
`RAG.retrieval`'s module docstring), on top of (not instead of) `RAG.retrieval`'s own
per-text SQLite cache for hosted embedding models, so a restart of `chatbot_server.py`
doesn't re-embed all 1,272 chunks every time. It's keyed by a hash of the model name and
every chunk's text, so it's invalidated automatically if the corpus or `RAG_EMBEDDING_MODEL`
changes.

**Command line:**

```bash
PYTHONPATH=dylan_pipeline python -m RAG.chatbot            # interactive chat
PYTHONPATH=dylan_pipeline python -m RAG.chatbot --demo     # scripted scenario-analysis walkthrough
```

**Browser UI**, served alongside the results dashboard (this is the one part of the project
that needs a running server — retrieval and any LLM call happen in Python, not the browser):

```bash
PYTHONPATH=dylan_pipeline python -m RAG.chatbot_server
```

then open `http://localhost:8001/chatbot.html` for the chat UI (with a "Scenario analysis
demo" tab showing the same walkthrough as `--demo`, as 13 cards with HIT/MISS badges), or
`http://localhost:8001/index.html` for the dashboard — `src/chatbot_server.py` serves
`dashboard/` as static files plus `/api/chat`, `/api/demo`, and `/api/info`, and the two
pages link to each other.

## Render fictional filings as PDFs (optional)

```bash
python -m src.step10_build_pdfs
```

- Writes `data/pdf_reports/{fictional_id}_{fy}.pdf` (git-ignored, derived): a disclaimer cover
  page plus Items 1/1A/1C/7/8, with the narrative's `label | value` lines rendered as real tables.
  Not part of the Phase 1-9 pipeline; built for sharing individual filings outside the repo (e.g.
  dragging them into a shared Google Drive folder). Requires `pandoc` and `xelatex` on `PATH`.

## Notes

- Fiscal year is the year of the filing's period of report. Non-December year-ends (several
  companies in the list) are labeled by the year their fiscal year ends.
- Per filing, the raw files are:
  - `10k.htm`: the 10-K exactly as filed, with inline XBRL tags. Opens in a browser.
  - `10k.txt`: readable plain text. Tables become `cell | cell` lines, and the hidden XBRL
    header is removed. This is the input for section splitting.
  - `ex13_*.htm`: Exhibit 13 (annual report), saved only when the filing includes one. Some
    banks put MD&A and financial statements there instead of in the 10-K body.
- `metadata.json` is written last. A folder counts as complete only when it exists.
- The full SEC submission (`.txt` with all exhibits and images) is no longer saved. It was
  45-75 MB per filing, mostly images and XBRL data files.
- Git tracks `metadata.json` files, `data/download_status.csv`, `data/parse_status.csv`,
  `data/companyfacts_status.csv`, `data/calibration.json`, `data/synthetic/`,
  `data/validation_report.csv`, `data/qa_pairs.csv`, `data/retrieval_eval.csv`,
  `data/gap_report.csv`, and `data/privacy_report.csv`. The filing and
  section text (`10k.htm`, `10k.txt`, `ex13_*`, `data/parsed/`), `data/companyfacts/*.json`,
  `data/index.csv`, `data/pdf_reports/`, and `logs/` are git-ignored, and are all reproducible by
  re-running the matching script.
- See `PLAN.md` for the full plan and open questions, and `SECTIONS.md` for which 10-K items
  are parsed and why.
