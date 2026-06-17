#!/usr/bin/env bash
# Start Ollama and optionally pull the small-model weights.
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b-instruct}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
SKIP_OLLAMA="${SKIP_OLLAMA:-0}"
SKIP_OLLAMA_PULL="${SKIP_OLLAMA_PULL:-0}"

if [[ "${SKIP_OLLAMA}" == "1" ]]; then
  echo "SKIP_OLLAMA=1 — Ollama startup skipped."
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it in the Dockerfile or set SKIP_OLLAMA=1."
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

echo "Small-model defaults:"
echo "  SMALL_MODEL_BASE_URL=http://${OLLAMA_HOST}/v1"
echo "  SMALL_MODEL_NAME=${OLLAMA_MODEL}"
