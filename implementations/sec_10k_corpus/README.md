# Real 10-K Corpus (SEC EDGAR)

Downloads and parses real US bank 10-K filings. This corpus is the real-data side of a
synthetic-10-K benchmark: it provides style templates, a comparison set, and XBRL facts.

**Status:** Step 1 (project setup) and step 2 (10-K download) are implemented. Section parsing,
the metadata index, XBRL companyfacts, and the summary script come next.

## Layout

```
config.yaml              SEC identity placeholders, company lists, fiscal years (committed)
config.local.yaml        your real SEC identity (git-ignored, create it yourself)
requirements.txt         edgartools, PyYAML, BeautifulSoup, lxml (pinned)
src/
  config.py              config loading, shared paths, identity check
  utils.py               logging setup, retry with backoff
  download_10k.py        step 2: download 10-K filings
  html_to_text.py        convert 10-K HTML to readable plain text
data/
  raw/{ticker}/{fy}/     10k.htm, 10k.txt, ex13_*.htm (if any), metadata.json
  parsed/                section files (step 3, not yet implemented)
  download_status.csv    one row per company-year from the last run
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

## Notes

- Fiscal year is the year of the filing's period of report. All banks in the list have
  December year-ends.
- Per filing, the raw files are:
  - `10k.htm`: the 10-K exactly as filed, with inline XBRL tags. Opens in a browser.
  - `10k.txt`: readable plain text. Tables become `cell | cell` lines, and the hidden XBRL
    header is removed. This is the input for section splitting.
  - `ex13_*.htm`: Exhibit 13 (annual report), saved only when the filing includes one. Many
    regional banks put MD&A and financial statements there instead of in the 10-K body.
- `metadata.json` is written last. A folder counts as complete only when it exists.
- The full SEC submission (`.txt` with all exhibits and images) is no longer saved. It was
  45-75 MB per filing, mostly images and XBRL data files.
- Git tracks `metadata.json` files and `data/download_status.csv`. The filing text
  (`10k.htm`, `10k.txt`, `ex13_*`) and `data/parsed/`, `data/index.csv`, and `logs/` are
  git-ignored. Re-running `python -m src.download_10k` recreates the filing text.
- See `PLAN.md` for the full plan and open questions, and `SECTIONS.md` for which 10-K items
  are parsed and why.
