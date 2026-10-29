# Generated batches

One folder per batch, named after the model and size (`gemini20`, `gpt4o20`, `opus20`, `claude20`).
`run_batch.sh` writes a new batch here with `RUN=<name>`.

```
data/<batch>/
  method_1/   Method 1's run: calibration.json, financials.csv, index.csv, parsed/ (symlinks), synthetic/filings/
  method_2/   Method 2's run: synthetic/<company>/<year>/item_*.txt, validation_report.csv
  evals/      that batch's eval reports (each method's checks on both batches)
```

Only `gemini20/method_1` and the `evals/` folders are in git. The other method folders are
generated on the Coder workspace and stay there. Comparisons across batches (the per-section
scorecard, compare_sets) stay in `evals/results/`. Pipeline inputs (real filings, XBRL
companyfacts, embedding cache) stay in each method's own `data/`.
