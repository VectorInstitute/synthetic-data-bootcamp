# Real 10-K Corpus (SEC EDGAR)

Downloads and parses real US bank 10-K filings. This corpus is the real-data side of a
synthetic-10-K benchmark: it provides style templates, a comparison set, and XBRL facts.

**Status:** Steps 1-3 (project setup, 10-K download, section parsing) are implemented. The
metadata index, XBRL companyfacts, and the summary script come next.

## Layout

```
config.yaml              SEC identity placeholders, company lists, fiscal years (committed)
config.local.yaml        your real SEC identity (git-ignored, create it yourself)
requirements.txt         edgartools, PyYAML, BeautifulSoup, lxml (pinned)
src/
  config.py              config loading, shared paths, identity check
  utils.py               logging setup, retry with backoff, atomic file writes
  download_10k.py        step 2: download 10-K filings
  parse_sections.py      step 3: split 10-K text into Items 1, 1A, 1C, 7, 8
  html_to_text.py        convert 10-K HTML to readable plain text
data/
  raw/{ticker}/{fy}/     10k.htm, 10k.txt, ex13_*.htm (if any), metadata.json
  parsed/{ticker}/{fy}/  item_1.txt, item_1a.txt, item_1c.txt, item_7.txt, item_8.txt
  download_status.csv    one row per company-year from the last download run
  parse_status.csv       one row per company-year-item from the last parse run
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

- `run_profile` in `config.yaml` selects the company list: `test` (JPM, PNC, FITB, KEY, ZION)
  or `full` (50 US bank holding companies). Set `fiscal_years` to change the years.
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
- For PGR, USB, and WFC, Item 7 and/or Item 8 are only "incorporated by reference" to an EX-13
  annual report exhibit. The heading named in that cross-reference sentence is located inside
  the matching `ex13_*.htm` instead, and used in place of the cross-reference text.
- Always reparses everything in `config.yaml`'s company list and fiscal years; it's local and
  fast, so there's no skip-if-done logic to go stale.
- Every section, found or not, gets a row in `parse_status.csv` (`ok`, `short`, or `missing`,
  plus character count and extraction method). Review `short` and `missing` rows before relying
  on a given section.

## Notes

- Fiscal year is the year of the filing's period of report. Non-December year-ends (Visa,
  Paychex, and a few others) are labeled by the year their fiscal year ends.
- Per filing, the raw files are:
  - `10k.htm`: the 10-K exactly as filed, with inline XBRL tags. Opens in a browser.
  - `10k.txt`: readable plain text. Tables become `cell | cell` lines, and the hidden XBRL
    header is removed. This is the input for section splitting.
  - `ex13_*.htm`: Exhibit 13 (annual report), saved only when the filing includes one. Many
    regional banks put MD&A and financial statements there instead of in the 10-K body.
- `metadata.json` is written last. A folder counts as complete only when it exists.
- The full SEC submission (`.txt` with all exhibits and images) is no longer saved. It was
  45-75 MB per filing, mostly images and XBRL data files.
- Git tracks `metadata.json` files, `data/download_status.csv`, and `data/parse_status.csv`.
  The filing and section text (`10k.htm`, `10k.txt`, `ex13_*`, `data/parsed/`) and `logs/` are
  git-ignored. Re-running `python -m src.download_10k` and `python -m src.parse_sections`
  recreates them.
- See `PLAN.md` for the full plan and open questions, and `SECTIONS.md` for which 10-K items
  are parsed and why.
