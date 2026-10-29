#!/usr/bin/env bash
# Generate one batch of synthetic 10-Ks with each method, using the same model and the
# same sections (Items 1, 1A, 1C, 7, 8), then run every eval on both batches.
#
# Run from the repository root on the Coder workspace, where Vector's proxy key is set:
#   bash implementations/synthetic_10k/run_batch.sh            # both methods + evals
#   bash implementations/synthetic_10k/run_batch.sh method1    # or: method2, evals
#
# Both methods use the repository's single uv environment (run `uv sync` once from the root).
#
# Settings (environment variables):
#   RUN        batch name, used for output folders            (default gemini20)
#   N          fictional companies per method               (default 20)
#   SEED       spec sampling seed                             (default 7)
#   MODEL      model on Vector's proxy                        (default gemini-3-flash-preview)
#   WORKERS    companies generated in parallel                (default 4)
#   EFFORT     Claude models only: thinking effort             (default high)
# A claude-* MODEL goes to the Anthropic API when ANTHROPIC_API_KEY is set (e.g. in the
# repo-root .env), otherwise to Vector's proxy like every other model.
# Method 2's step08 gets --max-tokens 16384: on Gemini the limit also covers hidden reasoning
# tokens, and the default 4096 cuts sections off mid-sentence.
#   M1_DATA  Method 1's parsed corpus with calibration.json      (default data/sec10k)
#
# Outputs:
# Outputs all go to one folder per batch, data/$RUN/:
#   data/$RUN/method_1/synthetic/filings/*/filing.html   Method 1's batch
#   data/$RUN/method_2/synthetic/*/*/item_*.txt          Method 2's batch
#   data/$RUN/evals/                                     all eval reports
set -euo pipefail

RUN=${RUN:-gemini20}
N=${N:-20}
SEED=${SEED:-7}
MODEL=${MODEL:-gemini-3-flash-preview}
WORKERS=${WORKERS:-4}
EFFORT=${EFFORT:-high}
ITEMS=(1 1A 1C 7 8)

REPO=$(pwd)
HERE="$REPO/implementations/synthetic_10k"
M1_DATA=$(realpath "${M1_DATA:-data/sec10k}")
M1_RUN="$HERE/data/$RUN/method_1"
M2_RUN="$HERE/data/$RUN/method_2"
RESULTS="$HERE/data/$RUN/evals"

# Make the proxy key in a repo-root .env visible to both methods.
if [[ -f "$REPO/.env" ]]; then set -a; source "$REPO/.env"; set +a; fi

method1() {
  for f in calibration.json index.csv financials.csv; do
    [[ -f "$M1_DATA/$f" ]] || { echo "Missing $M1_DATA/$f: run method_1.build and method_1.calibrate first (see method_1/README.md)"; exit 1; }
  done
  mkdir -p "$M1_RUN"
  for f in calibration.json index.csv financials.csv parsed; do ln -sfn "$M1_DATA/$f" "$M1_RUN/$f"; done
  cd "$HERE"
  uv run python -m method_1.specs --data "$M1_RUN" --n "$N" --seed "$SEED"
  local effort=()
  [[ "$MODEL" == claude-* ]] && effort=(--reasoning-effort "$EFFORT")
  uv run python -m method_1.narrative --data "$M1_RUN" --model "$MODEL" --items "${ITEMS[@]}" --workers "$WORKERS" "${effort[@]}"
  cd "$REPO"
}

method2() {
  cd "$HERE"
  if ! ls method_2/data/companyfacts/*.json >/dev/null 2>&1; then
    echo "Missing method_2/data/companyfacts/: set your SEC identity in method_2/config.local.yaml and run steps 01, 02 and 04 (see method_2/README.md), or copy an existing data/companyfacts and data/parsed there."
    exit 1
  fi
  ls method_2/data/parsed/*/ >/dev/null 2>&1 || echo "Note: method_2/data/parsed/ is empty, so prompts get no real style exemplar."
  export SYNTHETIC_DIR="$M2_RUN/synthetic"
  uv run python -m method_2.step06_make_specs --n "$N" --seed "$SEED"
  uv run python -m method_2.step07_make_financials
  local backend=openai
  [[ "$MODEL" == claude-* && -n "${ANTHROPIC_API_KEY:-}" ]] && backend=anthropic
  uv run python -m method_2.step08_generate_filings --backend "$backend" --model "$MODEL" --workers "$WORKERS" --max-tokens 16384 --effort "$EFFORT"
  uv run python -m method_2.step09_validate_filings
  unset SYNTHETIC_DIR
  cd "$REPO"
}

evals() {
  mkdir -p "$RESULTS"
  cd "$HERE"
  # Method 2's batch: its own gates, then Method 1's checks.
  cp "$M2_RUN/validation_report.csv" "$RESULTS/method2_gates_on_method2.csv"
  uv run python -m evals.method1_checks --synthetic-dir "$M2_RUN/synthetic" --out "$RESULTS/method1_checks_on_method2.csv"
  # Method 1's batch: its own checks (recorded during generation), then Method 2's gates.
  cp "$(ls -t "$M1_RUN"/synthetic/runs/*.json | head -1)" "$RESULTS/method1_checks_on_method1.json"
  uv run python -m evals.method2_gates "$M1_RUN"/synthetic/filings/*/filing.html --out-dir "$RESULTS/method2_gates_on_method1"
  cd "$REPO"
  echo "Eval reports in $RESULTS"
}

case "${1:-all}" in
  method1) method1 ;;
  method2) method2 ;;
  evals) evals ;;
  all) method1; method2; evals ;;
  *) echo "usage: $0 [all|method1|method2|evals]"; exit 2 ;;
esac
