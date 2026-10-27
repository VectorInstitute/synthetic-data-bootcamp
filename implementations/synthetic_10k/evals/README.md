# Evals across both pipelines

Each script runs one pipeline's validator on the other pipeline's filings without changing the validator. Failures carry a `note` column that says when a failure is a difference between the two validators rather than a defect in the filing.

| Script | What it runs | On | Output |
|---|---|---|---|
| `phase7_gates.py` | Dylan's Phase 7 gates (`dylan_pipeline/src/step09_validate_filings.py`) | `filing.html` files from Amin's pipeline; the two showcase filings are in `samples/amin_showcase/` | `results/phase7_on_amin_showcase/`: each filing converted into Dylan's layout, plus `validation_report.csv` |
| `amin_checks.py` | Amin's narrative checks (`amin_pipeline/sec10k/validate.py`) | Dylan's 20 fictional companies in `dylan_pipeline/data/synthetic/` | `results/amin_checks_on_dylan20/validation_report.csv` |

Dylan's own run of his gates on his filings is in `dylan_pipeline/data/validation_report.csv`. The [folder overview](../README.md#evals) summarises all three results.

## Run

From this folder, in an environment with Dylan's requirements (`../dylan_pipeline/requirements.txt`) and the repo's `uv` environment:

```bash
python phase7_gates.py samples/amin_showcase/*.html
python amin_checks.py
```

To check new filings from Amin's pipeline, pass their `filing.html` paths (written to `data/sec10k/synthetic/filings/{company_id}/`) to `phase7_gates.py`, with `--out-dir` set to a new folder under `results/`. `amin_checks.py` always checks every company under `dylan_pipeline/data/synthetic/`.

## What each script adapts

- `phase7_gates.py`: Amin's filings show two fiscal years and don't record which real company was the style example. The script accepts prior-year values, checks all 503 S&P 500 names instead of one template, and notes year-over-year figures that Dylan's single-year check doesn't expect. Details are in its docstring.
- `amin_checks.py`: runs the checks that don't depend on Amin's prompt format: company name, real names, number density, leaked variable names, numbers, and overlap with the template filing (only where Dylan's git-ignored `data/parsed/` exists). Dylan's single-year statements are used as both years.
