# Real 10-K Corpus (SEC EDGAR)

Downloads and parses real 10-K filings for 50 randomly sampled S&P 500 companies. This corpus
is the real-data side of a synthetic-10-K benchmark: it provides style templates, a comparison
set, and XBRL facts.

**Status:** Steps 1-5 (project setup, 10-K download, section parsing, metadata index, XBRL
companyfacts) and Phases 3-4 (calibration profile, fictional-company specs) are implemented. The
summary script and synthetic filing generation (Phase 5 onward) come next; see `PLAN.md`.

## Layout

```
config.yaml              SEC identity placeholders, company lists, fiscal years (committed)
config.local.yaml        your real SEC identity (git-ignored, create it yourself)
requirements.txt         edgartools, PyYAML, BeautifulSoup, lxml (pinned)
src/
  config.py              config loading, shared paths, identity check
  utils.py               logging setup, retry with backoff, atomic file writes
  sample_companies.py    reproducibly sample config.yaml's 50-ticker company list
  download_10k.py        step 2: download 10-K filings
  parse_sections.py      step 3: split 10-K text into Items 1, 1A, 1C, 7, 8
  html_to_text.py        convert 10-K HTML to readable plain text
  build_index.py         step 4: data/index.csv, one row per parsed section
  fetch_companyfacts.py  step 5: download XBRL companyfacts per company
  make_calibration.py    Phase 3: data/calibration.json from companyfacts
  make_specs.py          Phase 4: fictional-company specs from calibration.json
data/
  sp500_constituents.csv frozen S&P 500 snapshot src.sample_companies samples from (committed)
  raw/{ticker}/{fy}/     10k.htm, 10k.txt, ex13_*.htm (if any), metadata.json
  parsed/{ticker}/{fy}/  item_1.txt, item_1a.txt, item_1c.txt, item_7.txt, item_8.txt
  companyfacts/{ticker}.json   raw XBRL companyfacts per company (git-ignored, ~180 MB)
  calibration.json       percentile distributions and per-company-year records (committed)
  index.csv              one row per parsed section (git-ignored, reproduce when needed)
  download_status.csv    one row per company-year from the last download run
  parse_status.csv       one row per company-year-item from the last parse run
  companyfacts_status.csv one row per company from the last companyfacts fetch
  synthetic/{fictional_id}/{fy}/spec.json   fictional-company specs (committed)
logs/                    per-script log files
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
python -m src.download_10k
```

- `run_profile` in `config.yaml` selects the company list: `test` (JPM, PNC, FITB, KEY, ZION, a
  small sanity-check set) or `full` (50 S&P 500 companies sampled with `src.sample_companies`;
  see its docstring for how and why). Set `fiscal_years` to change the years.
- Re-running skips any company-year whose folder is already complete.
- Missing company-years (no 10-K for that fiscal year, or ticker not found) are logged and
  recorded in `download_status.csv`. The run continues.
- Rate limiting: edgartools throttles requests to 9 per second by default, under SEC's
  10/second limit. Network errors and HTTP 429 responses are retried with backoff by edgartools.
  The download script adds another retry layer for other failures.

## Parse 10-K sections

```bash
python -m src.parse_sections
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
python -m src.build_index
python -m src.fetch_companyfacts
```

- `build_index.py` writes `data/index.csv`, one row per parsed section (ticker, company name,
  CIK, fiscal year, filing date, accession number, item, path, character count, word count).
  Local and fast; always rewrites the whole file.
- `fetch_companyfacts.py` downloads each company's XBRL companyfacts JSON from
  `data.sec.gov/api/xbrl/companyfacts/`, using the CIK from its already-downloaded
  `metadata.json`. Re-running skips any ticker whose file already exists.

## Build the calibration profile

```bash
python -m src.make_calibration
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
python -m src.make_specs
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
- Deterministic: re-running regenerates the same 20 specs (fixed seed in `src/make_specs.py`).

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
  `data/companyfacts_status.csv`, `data/calibration.json`, and `data/synthetic/`. The filing and
  section text (`10k.htm`, `10k.txt`, `ex13_*`, `data/parsed/`), `data/companyfacts/*.json`, and
  `data/index.csv` and `logs/` are git-ignored, and are all reproducible by re-running the
  matching script.
- See `PLAN.md` for the full plan and open questions, and `SECTIONS.md` for which 10-K items
  are parsed and why.
