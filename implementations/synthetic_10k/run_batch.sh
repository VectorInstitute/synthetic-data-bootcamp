#!/usr/bin/env bash
# Generate one batch of synthetic 10-Ks with each pipeline, using the same model and the
# same sections (Items 1, 1A, 1C, 7, 8), then run every eval on both batches.
#
# Run from the repository root on the Coder workspace, where Vector's proxy key is set:
#   bash implementations/synthetic_10k/run_batch.sh            # both pipelines + evals
#   bash implementations/synthetic_10k/run_batch.sh amin       # or: dylan, evals
#
# Settings (environment variables):
#   RUN        batch name, used for output folders            (default gemini20)
#   N          fictional companies per pipeline               (default 20)
#   SEED       spec sampling seed                             (default 7)
#   MODEL      model on Vector's proxy                        (default gemini-3-flash-preview)
#   WORKERS    companies generated in parallel                (default 4)
# Dylan's step08 gets --max-tokens 16384: on Gemini the limit also covers hidden reasoning
# tokens, and the default 4096 cuts sections off mid-sentence.
#   AMIN_DATA  Amin's parsed corpus with calibration.json      (default data/sec10k)
#
# Outputs:
#   amin_pipeline/data/runs/$RUN/synthetic/filings/*/filing.html   Amin's batch
#   dylan_pipeline/data/runs/$RUN/synthetic/*/*/item_*.txt         Dylan's batch
#   evals/results/$RUN/                                            all eval reports
set -euo pipefail

RUN=${RUN:-gemini20}
N=${N:-20}
SEED=${SEED:-7}
MODEL=${MODEL:-gemini-3-flash-preview}
WORKERS=${WORKERS:-4}
ITEMS=(1 1A 1C 7 8)

REPO=$(pwd)
HERE="$REPO/implementations/synthetic_10k"
AMIN_DATA=$(realpath "${AMIN_DATA:-data/sec10k}")
AMIN_RUN="$HERE/amin_pipeline/data/runs/$RUN"
DYLAN_RUN="$HERE/dylan_pipeline/data/runs/$RUN"
RESULTS="$HERE/evals/results/$RUN"

# Make the proxy key in a repo-root .env visible to both pipelines.
if [[ -f "$REPO/.env" ]]; then set -a; source "$REPO/.env"; set +a; fi

amin() {
  for f in calibration.json index.csv financials.csv; do
    [[ -f "$AMIN_DATA/$f" ]] || { echo "Missing $AMIN_DATA/$f: run sec10k.build and sec10k.calibrate first (see amin_pipeline/README.md)"; exit 1; }
  done
  mkdir -p "$AMIN_RUN"
  for f in calibration.json index.csv financials.csv parsed; do ln -sfn "$AMIN_DATA/$f" "$AMIN_RUN/$f"; done
  cd "$HERE/amin_pipeline"
  uv run python -m sec10k.specs --data "$AMIN_RUN" --n "$N" --seed "$SEED"
  uv run python -m sec10k.narrative --data "$AMIN_RUN" --model "$MODEL" --items "${ITEMS[@]}" --workers "$WORKERS"
  cd "$REPO"
}

dylan() {
  cd "$HERE/dylan_pipeline"
  [[ -d .venv ]] || uv venv --python 3.12 .venv
  uv pip install --quiet --python .venv -r requirements.txt
  if ! ls data/companyfacts/*.json >/dev/null 2>&1; then
    echo "Missing data/companyfacts/: set your SEC identity in config.local.yaml and run steps 01, 02 and 04 (see dylan_pipeline/README.md), or copy Dylan's data/companyfacts and data/parsed here."
    exit 1
  fi
  ls data/parsed/*/ >/dev/null 2>&1 || echo "Note: data/parsed/ is empty, so prompts get no real style exemplar."
  export SYNTHETIC_DIR="$DYLAN_RUN/synthetic"
  .venv/bin/python -m src.step06_make_specs --n "$N" --seed "$SEED"
  .venv/bin/python -m src.step07_make_financials
  .venv/bin/python -m src.step08_generate_filings --backend openai --model "$MODEL" --workers "$WORKERS" --max-tokens 16384
  .venv/bin/python -m src.step09_validate_filings
  unset SYNTHETIC_DIR
  cd "$REPO"
}

evals() {
  mkdir -p "$RESULTS"
  local py="$HERE/dylan_pipeline/.venv/bin/python"
  cd "$HERE/evals"
  # Dylan's batch: his own gates, then Amin's checks.
  cp "$DYLAN_RUN/validation_report.csv" "$RESULTS/dylan_gates_on_dylan.csv"
  "$py" amin_checks.py --synthetic-dir "$DYLAN_RUN/synthetic" --out "$RESULTS/amin_checks_on_dylan.csv"
  # Amin's batch: his own checks (recorded during generation), then Dylan's gates.
  cp "$(ls -t "$AMIN_RUN"/synthetic/runs/*.json | head -1)" "$RESULTS/amin_checks_on_amin_run.json"
  "$py" phase7_gates.py "$AMIN_RUN"/synthetic/filings/*/filing.html --out-dir "$RESULTS/dylan_gates_on_amin"
  cd "$REPO"
  echo "Eval reports in $RESULTS"
}

case "${1:-all}" in
  amin) amin ;;
  dylan) dylan ;;
  evals) evals ;;
  all) amin; dylan; evals ;;
  *) echo "usage: $0 [all|amin|dylan|evals]"; exit 2 ;;
esac
