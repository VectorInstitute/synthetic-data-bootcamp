# Domain-Specific QA Alignment

Reference implementation for improving small-language-model (SLM) alignment with
synthetic, domain-specific Q&A. The starter domain is finance policy documents;
the same notebooks can be adapted to other document types.

## Domain (finance)

| Document | Role | Purpose |
|----------|------|---------|
| CFPB credit card agreement | Policy-dense | Format, vocabulary, multi-constraint questions |
| SEC investor bulletin | Scope-boundary | Refusal vs engagement calibration |

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment management
- OpenAI-compatible API key for teacher and judge models (see [`.env.example`](./.env.example))
- [Ollama](https://ollama.com/) when you want a local SLM for baseline / eval (optional until those steps)

## Data layout

Generated under `data/` as you run the notebooks:

```
data/
├── documents/          # source policy text (sample excerpts included)
├── paragraphs.jsonl    # chunked paragraphs with train/test split (notebook 01)
├── test/
│   └── test_set.jsonl  # held-out evaluation Q&A (notebook 01)
├── synthetic/
│   ├── synthetic_raw.jsonl       # notebook 02
│   ├── synthetic_filtered.jsonl  # notebook 03
│   └── synthetic_train.jsonl     # notebook 04
└── results/
    ├── baseline_predictions.jsonl  # notebook 01
    ├── baseline_scores.json        # notebook 01
    ├── finetuned_predictions.jsonl # notebook 05
    └── comparison_report.json      # notebook 05
```

## Notebooks

Run in order. Each notebook is a tutorial; reusable logic lives in
`aieng-synthetic-data/aieng/syn_data/text/`.

| Notebook | Step | Description |
|----------|------|-------------|
| [01_baseline_evaluation.ipynb](./01_baseline_evaluation.ipynb) | 1 | Build test set, run baseline, LLM-as-judge |
| [02_synthetic_qa_generation.ipynb](./02_synthetic_qa_generation.ipynb) | 2 | Compare generation strategies (teacher LLM) |
| [03_quality_filtering.ipynb](./03_quality_filtering.ipynb) | 3 | Heuristic filters + judge scoring |
| [04_grounded_data_augmentation.ipynb](./04_grounded_data_augmentation.ipynb) | 4 | Instruction back-translation SFT corpus |
| [05_finetune_and_compare.ipynb](./05_finetune_and_compare.ipynb) | 5 | LoRA SFT + before/after comparison |

## Package

Shared utilities are provided by the workspace package `aieng-synthetic-data`:

```python
from aieng.syn_data.text import (
    list_domain_documents,
    sample_test_paragraphs,
    zero_shot_generate,
    apply_heuristic_filters,
    generate_grounded_qagenerate_instruction_back_translation_sample,
    run_inference,
)
```

## Getting started

### 1. Coder workspace setup (default for the bootcamp)

1. Sync dependencies from the **repository root** (installs the text-SFT group used by this RI, including libraries such as `json_repair`, `openai`, `rich`, and optional SFT stacks when enabled):

```bash
uv sync --dev --group text-sft
```

2. Copy env defaults into this directory and set your API key:

```bash
# from implementations/qa_text_generation/
cp .env.example .env   # then set OPENAI_API_KEY (and adjust models if needed)
```

Notebooks load this file via `load_implementation_dotenv()` (path is fixed to this RI, so it works after the notebook `chdir`s to the repo root).

3. **Start Ollama manually** when you need the local SLM (baseline inference, eval, trying other small models). This is an educational step: deploy an SLM on CPU (or GPU if available) with Ollama. Installation may already be present in the Docker/Coder image; starting the server is intentional and not tied to image boot:

```bash
# from repository root
bash scripts/start-ollama.sh
```

That script reads `OLLAMA_*` / `SMALL_MODEL_*` from this RI’s `.env` (if present), starts the server if needed, and pulls the configured model.

To try a different SLM, update **both** of these in `.env` (keep them equal), then re-run `start-ollama.sh`:

| Variable | Role |
|----------|------|
| `OLLAMA_MODEL` | Model tag pulled/served by Ollama (e.g. `qwen2.5:3b-instruct`) |
| `SMALL_MODEL_NAME` | Name passed to the OpenAI-compatible client |
| `OLLAMA_HOST` | Host:port for the Ollama server (default `127.0.0.1:11434`) |
| `SMALL_MODEL_BASE_URL` | Same host with `/v1` (Ollama’s OpenAI-compatible API) |

Optional local cleanup: `ollama stop qwen2.5:3b-instruct` (Coder VMs usually tear this down on stop anyway).

4. Open `01_baseline_evaluation.ipynb` and work through notebooks 01 → 05.

Notebook 05 runs LoRA fine-tuning only when `RUN_SFT=1` (CUDA). Without that flag it still runs the comparison workflow using the small-model client.

To confirm CUDA on a GPU workspace:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **TODO:** Confirm with Amrit whether participants use the Vector proxy for API keys.

### 2. Local development (skip on Coder)

Same env + notebook flow as above. Install Ollama yourself if you want the local SLM; otherwise you can skip `start-ollama.sh` and only run teacher/judge steps that use the cloud API (baseline SLM cells will fail without a small-model endpoint).

```bash
# from repository root
uv sync --dev --group text-sft
cp implementations/qa_text_generation/.env.example implementations/qa_text_generation/.env
# edit .env — set OPENAI_API_KEY
bash scripts/start-ollama.sh   # optional until you need the SLM
```

### 3. Docker + GCP VM (optional)

The Dockerfile **installs** Ollama but does **not** start it at boot. After Jupyter is up, start it yourself with `bash scripts/start-ollama.sh`.

```bash
docker build -t synthetic-data-bootcamp .
docker run --rm -p 8888:8888 \
  -e OPENAI_API_KEY=vp-... \
  -e OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1 \
  -v ollama-models:/home/coder/.ollama \
  synthetic-data-bootcamp
```

Open Jupyter at `http://localhost:8888`, copy `.env` under this RI if needed, then start Ollama as above. On first start, pulling `qwen2.5:3b-instruct` may take a few minutes (~2 GB).

**GCP VM (CPU, e.g. `e2-standard-4`):**

1. Create a VM with ≥30 GB boot disk (model cache + repo).
2. Install Docker, clone the repo, build the image.
3. Run the container with port `8888` open in the firewall.
4. Mount a persistent volume on `/home/coder/.ollama` so the model is not re-downloaded every restart.
5. Keep teacher/judge on the cloud API; only the small model runs locally via Ollama.

**Optional GPU VM (NVIDIA only):** LoRA SFT in notebook 05 needs **NVIDIA CUDA** (not Apple Silicon / Metal).

```bash
docker run --rm --gpus all -p 8888:8888 \
  -e RUN_SFT=1 \
  -e OPENAI_API_KEY=vp-... \
  -e OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1 \
  -v ollama-models:/home/coder/.ollama \
  synthetic-data-bootcamp
```

Then start Ollama manually inside the environment and verify CUDA:

```bash
bash scripts/start-ollama.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
