# Policy-Document QA Alignment

Reference implementation for improving small-model alignment with synthetic,
domain-specific Q&A generated from company policy documents.

## Domain (finance)

| Document | Role | Purpose |
|----------|------|---------|
| CFPB credit card agreement | Policy-dense | Format, vocabulary, multi-constraint questions |
| SEC investor bulletin | Scope-boundary | Refusal vs engagement calibration |

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment management
- OpenAI-compatible API key for teacher and judge models (see `.env.example`)
- Ollama

```bash
# from repository root
uv sync --dev
cp .env.example .env  # then add your API key
```

## Data layout

```
data/
├── documents/          # source policy text (sample excerpts included)
├── paragraphs.jsonl    # chunked paragraphs with train/test split (Step 1)
├── test/
│   └── test_set.jsonl  # held-out evaluation Q&A (Step 1)
├── synthetic/
│   ├── synthetic_raw.jsonl
│   ├── synthetic_filtered.jsonl
│   └── synthetic_train.jsonl
└── results/
    ├── baseline_predictions.jsonl
    ├── baseline_scores.json
    └── comparison_report.json
```

## Notebooks

Run in order. Each notebook is a tutorial; reusable logic lives in
`aieng-synthetic-data/aieng/syn_data/text/`.

| Notebook | Step | Description |
|----------|------|-------------|
| [01_baseline_evaluation.ipynb](./01_baseline_evaluation.ipynb) | 1 | Build test set, run baseline, LLM-as-judge |
| [02_synthetic_qa_generation.ipynb](./02_synthetic_qa_generation.ipynb) | 2 | Compare generation strategies (teacher LLM) |
| [03_quality_filtering.ipynb](./03_quality_filtering.ipynb) | 3 | Heuristic filters + judge scoring |
| [04_grounded_data_augmentation.ipynb](./04_grounded_data_augmentation.ipynb) | 4 | RAG-grounded SFT corpus (500–1k) |
| [05_finetune_and_compare.ipynb](./05_finetune_and_compare.ipynb) | 5 | LoRA SFT + before/after comparison |

## Package

Shared utilities are provided by the workspace package `aieng-synthetic-data`:

```python
from aieng.syn_data.text import (
    list_domain_documents,
    sample_test_paragraphs,
    zero_shot_generate,
    apply_heuristic_filters,
    generate_grounded_qa,
    run_inference,
)
```

## Getting started

1. Sync the environment from the repository root.
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (and optional `SMALL_MODEL_*`).
3. Open `01_baseline_evaluation.ipynb` in JupyterLab.
4. Work through notebooks 01 → 05 in order.

Notebook 05 runs LoRA fine-tuning only when `RUN_SFT=1` on a CUDA machine. Without
that flag, it still completes the comparison workflow using the small-model client.

For GPU fine-tuning, install optional packages in your environment:

```bash
uv pip install torch transformers peft trl bitsandbytes accelerate datasets
```

## Docker + GCP (Ollama for the small model)

**For local development - skip on Coder**
The Dockerfile installs **Ollama** and starts it via `scripts/setup.sh`. On first boot
it pulls `qwen2.5:3b-instruct` (~2 GB). Teacher and judge still use your cloud API key.


```bash
docker build -t synthetic-data-bootcamp .
docker run --rm -p 8888:8888 \
  -e OPENAI_API_KEY=vp-... \
  -e OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1 \
  -v ollama-models:/home/coder/.ollama \
  synthetic-data-bootcamp
```

Open Jupyter at `http://localhost:8888` and run the notebooks. The small-model env
vars are pre-set to the local Ollama endpoint.

**GCP VM (CPU, e.g. `e2-standard-4`):**

1. Create a VM with ≥30 GB boot disk (model cache + repo).
2. Install Docker, clone the repo, build the image.
3. Run the container with port `8888` open in the firewall.
4. Mount a persistent disk/volume to `/home/coder/.ollama` so the model is not
   re-downloaded on every restart.
5. Keep teacher/judge on OpenAI; only the small model runs locally.

**Optional GPU VM (NVIDIA only):** LoRA SFT in notebook 05 requires **NVIDIA CUDA**, not
Apple Silicon / Metal. It will not work on macOS (including Docker Desktop on a Mac).

On a Linux VM with an NVIDIA GPU:

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
2. Pass the GPU into the container and enable SFT deps at boot:

```bash
docker run --rm --gpus all -p 8888:8888 \
  -e RUN_SFT=1 \
  -e OPENAI_API_KEY=vp-... \
  -e OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1 \
  -v ollama-models:/home/coder/.ollama \
  synthetic-data-bootcamp
```
Then verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Coder Workspace Setup

On Coder, the equivalent of step 2 is:

| Docker step | Coder native equivalent |
|-------------|---------------------------|
| `--gpus all` | Not needed — GPU is on the VM already |
| `RUN_SFT=1` | Set in `.env` (you already have this) |
| SFT deps at boot | `uv sync --dev --group text-sft` from repo root |

1. Install Ollama and optional dependcies for suprevised fine-tuning(SFT):
```sh
curl -fsSL https://ollama.com/install.sh | sh
bash scripts/start-ollama.sh
# optional, for notebook 05 SFT on GPU:
uv sync --dev --group text-sft   # with RUN_SFT=1 in .env
```

2. Inside the container (or in a notebook cell), confirm CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ollama will also use the GPU automatically when `--gpus all` is set.

**Skip Ollama** (e.g. local dev without the small model):

```bash
SKIP_OLLAMA=1 docker run ...
```
