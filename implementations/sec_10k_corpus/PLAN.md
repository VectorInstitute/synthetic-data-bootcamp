# Synthetic 10-K Dataset: Plan and Open Questions

Goal: build a synthetic 10-K dataset for evaluating RAG systems on equity-analysis questions.
Real SEC 10-K filings serve three roles: style templates, calibration for realistic numbers, and
a real-data benchmark.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Project setup, config, identity handling | Done |
| 2 | Download 10-K filings from EDGAR | Done: 50 companies x FY2022-2024 (150 company-years, 0 errors) |
| 3 | Parse filings into sections (Items 1, 1A, 1C, 7, 8) | Not started |
| 4 | Metadata index (`data/index.csv`) | Not started |
| 5 | XBRL companyfacts JSON per company | Not started |
| 6 | Summary script | Not started |
| - | Calibration, company specs, filing generation, QA, retrieval evaluation | Not started |

### Downloaded data

- Location: `data/raw/{ticker}/{fy}/`, with `10k.htm` (the 10-K as filed), `10k.txt` (readable text),
  `ex13_*.htm` (annual report exhibit, when present), and `metadata.json`.
- Companies: 15 US banks (JPM, BAC, WFC, C, USB, PNC, TFC, COF, MTB, FITB, KEY, RF, HBAN, CFG, ZION)
  plus 35 S&P 500 companies drawn at random with seed 42.
- EX-13 annual reports are present for PGR, USB, and WFC (all three years each). Most other filings
  keep MD&A and financials in the 10-K body.
- Size: about 1.3 GB of filing text, plus about 135 KB of metadata and the status file.
- Reproduce: `python -m src.download_10k` (skips anything already downloaded).

## Plan

### Phase 1: Parse filings into sections (step 3)

- Input: `10k.txt`. Output: `data/parsed/{ticker}/{fy}/item_1.txt`, `item_1a.txt`, `item_1c.txt`,
  `item_7.txt`, `item_8.txt`. See `SECTIONS.md` for which items are included and why.
- The table of contents repeats every Item heading. Pick the body occurrence, meaning the heading
  followed by the most text. Body headings are uppercase, for example `ITEM 1A – RISK FACTORS`.
- When Item 7 or 8 is only a cross-reference to EX-13, take the text from the matching `ex13_*.htm`.
- Item 1C exists only from FY2023 onward.
- Missing or suspiciously short sections go into an exceptions CSV for manual review.

### Phase 2: Index and XBRL facts (steps 4 and 5)

- `data/index.csv`: one row per section, with ticker, company name, CIK, fiscal year, filing date,
  accession number, item, path, character count, and word count.
- XBRL companyfacts: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`, one file per
  company, fetched through edgartools' `get_with_retry` so it uses the same identity and rate limit.

### Phase 3: Calibration profile

- From the companyfacts data, extract per bank-year: total assets, loans, deposits, equity, net
  income, CET1 ratio, net interest margin, and return on equity.
- Compute percentile distributions and save them as `calibration.json`.
- Calibration uses only the 15 banks. The 35 S&P 500 companies are not banks, so they don't
  calibrate bank specs. They are still useful as real filings for the benchmark.

### Phase 4: Company spec generator

- Each fictional bank is a JSON spec: fictional name, HQ state, fiscal year, size sampled from the
  calibration distributions, balance-sheet mix, capital ratios, loan concentrations (for example,
  commercial real estate share), footprint, employees and branches, and the cybersecurity facts
  that Item 1C needs.
- Reject specs that break accounting identities: assets equal liabilities plus equity, and loans do
  not exceed assets.
- Check every fictional name against the real company list, so no fictional bank carries a real
  name or ticker.

### Phase 5: Compute financial statements, don't write them

- Code builds the balance sheet, income statement, and cash-flow tables from the spec. Every table
  reconciles.
- The language model writes only narrative text. All numbers are fixed before any prompt.

### Phase 6: Narrative generation per Item

- Generation order: Item 1, then 1A, then 7 (which quotes the Item 8 numbers), then 1C, then the
  Item 8 notes. Earlier output is passed in as context so sections stay consistent.
- Each prompt contains: spec facts, computed numbers, one real section of the same Item as a style
  exemplar (not to be copied), a length target from the real distribution, and the real heading
  outline.
- Choose the model after a small pilot: a stronger model for MD&A, a cheaper one for volume.

### Phase 7: Validation gates

- Automatic: MD&A figures match the Item 8 tables, ratios match computed values, section lengths
  fall within the real distribution, required headings are present, and no real company names
  appear.
- Model-judged: realism rubric on a sample.
- Human: review about 10 random filings.
- A failing filing gets one regeneration. If it still fails, it is dropped and logged.

### Phase 8: Same layout as the real corpus

- `data/synthetic/{fictional_id}/{fy}/item_*.txt`, with index rows marked `source = synthetic`.
  The same index and retrieval code then works on both corpora.

### Phase 9: QA and retrieval evaluation

- Synthetic filings: the spec holds the true values, so QA pairs can have exact answers and gold
  passage locations.
- Real filings: generate questions with a model, then check that each answer span appears in the
  text.
- Retrievers: BM25, dense embeddings, and hybrid (reciprocal rank fusion).
- Metrics: recall@k, MRR, nDCG@10.
- Experiments: synthetic only, real only, and mixed.

### Phase 10: Synthetic-vs-real gap report

- Compare section lengths, vocabulary overlap, number distributions, and BM25 score distributions.
- Check whether the retriever ranking on synthetic matches the ranking on real. If it does,
  synthetic data is a valid proxy for the real benchmark. If not, the report says so.

### Command sequence (once built)

```
python -m src.download_10k          # done
python -m src.parse_sections        # Phase 1 (step 3)
python -m src.build_index           # Phase 2 (step 4)
python -m src.fetch_companyfacts    # Phase 2 (step 5)
python -m src.make_calibration      # Phase 3
python -m src.make_specs            # Phase 4
python -m src.generate_filings      # Phases 5 and 6
python -m src.validate_filings      # Phase 7
python -m src.make_qa               # Phase 9
python -m src.evaluate_retrievers   # Phase 9
```

### Risks

- **Verbatim leakage:** narrative built from real exemplars can echo real phrasing. Keep exemplars
  short and flag long n-gram overlaps with the source.
- **Uniform prose:** generated text is usually more regular than real filings. The gap report
  should say so.
- **XBRL gaps:** some banks do not report every tag, so calibration needs fallbacks.
- **Name impersonation:** fictional banks must not use real names or tickers.

## Open questions

1. **Synthetic corpus size:** how many synthetic banks, and which fiscal years? For example,
   20 banks for FY2023 and FY2024.
2. **Sections:** only Items 1, 1A, 1C, 7, and 8 as specified, or full 10-Ks?
3. **Model and budget:** which model, and what budget? Narrative cost scales with
   banks x sections.
4. **Calibration:** confirm that calibration uses only the 15 banks (recommended).
5. **Embedding model:** which model for the dense retriever?
6. **Sharing the raw data:** the 1.3 GB of filing text should not go into git. Options:
   - a GitHub Release asset (zip, up to 2 GB per file, no repo bloat), or
   - a shared drive or Hugging Face dataset, or
   - team members re-run `python -m src.download_10k` with the same config (it is deterministic).
   Which do you want?
7. **Repository visibility:** the GitHub repo is public. Is that intended for this project?
8. **Leftover folders:** 15 companies from the earlier, stopped regional-bank run still have
   folders in `data/raw/`. Delete them, or keep them?
9. **EX-13 handling:** the parser needs a special path for PGR, USB, and WFC. Is it OK to start
   with the 47 filings that have no EX-13 and add those three later?

## Milestones

- **M1:** parser runs on all 50 companies x 3 years, with an exceptions list.
- **M2:** index, companyfacts, and `calibration.json`.
- **M3:** spec generator and financial-statement computation, with reconciliation tests.
- **M4:** narrative pilot on 2 fictional banks, validators, and review.
- **M5:** full synthetic corpus.
- **M6:** QA, retrieval evaluation, and gap report.
