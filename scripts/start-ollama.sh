#!/usr/bin/env bash
# Start Ollama and optionally pull the small-model weights.
# Run manually from the repository root when you need the local SLM:
#   bash scripts/start-ollama.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/implementations/qa_text_generation/.env"

# Prefer RI .env so OLLAMA_* / SMALL_MODEL_* stay aligned with notebooks.
if [[ -f "${ENV_FILE}" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue
    if [[ "${line}" =~ ^(OLLAMA_|SKIP_OLLAMA|SMALL_MODEL_) ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      # Do not override vars already set in the shell / container env.
      if [[ -z "${!key:-}" ]]; then
        export "${key}=${value}"
      fi
    fi
  done < "${ENV_FILE}"
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-${SMALL_MODEL_NAME:-qwen2.5:0.5b-instruct}}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
SKIP_OLLAMA="${SKIP_OLLAMA:-0}"
SKIP_OLLAMA_PULL="${SKIP_OLLAMA_PULL:-0}"

if [[ "${SKIP_OLLAMA}" == "1" ]]; then
  echo "SKIP_OLLAMA=1 — Ollama startup skipped."
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it (e.g. curl -fsSL https://ollama.com/install.sh | sh)"
  echo "or set SKIP_OLLAMA=1."
  exit 1
fi

if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at http://${OLLAMA_HOST}."
else
  echo "Starting Ollama on http://${OLLAMA_HOST} ..."
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!
  echo "${OLLAMA_PID}" >/tmp/ollama.pid

  retries=30
  until curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; do
    retries=$((retries - 1))
    if [[ "${retries}" -le 0 ]]; then
      echo "Ollama failed to start. Log:"
      tail -n 50 /tmp/ollama.log || true
      exit 1
    fi
    sleep 2
  done
  echo "Ollama is ready."
fi

if [[ "${SKIP_OLLAMA_PULL}" == "1" ]]; then
  echo "SKIP_OLLAMA_PULL=1 — skipping model pull."
  exit 0
fi

if ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${OLLAMA_MODEL}"; then
  echo "Model '${OLLAMA_MODEL}' already present."
else
  echo "Pulling model '${OLLAMA_MODEL}' (first run may take several minutes) ..."
  ollama pull "${OLLAMA_MODEL}"
fi

echo "Small-model defaults (keep these aligned in .env):"
echo "  SMALL_MODEL_BASE_URL=http://${OLLAMA_HOST}/v1"
echo "  SMALL_MODEL_NAME=${OLLAMA_MODEL}"
echo "  OLLAMA_MODEL=${OLLAMA_MODEL}"
echo
echo "To stop this model later (local use): ollama stop ${OLLAMA_MODEL}"
