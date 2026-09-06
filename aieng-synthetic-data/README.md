# aieng-synthetic-data

Reusable utilities for the synthetic data bootcamp, focused on policy-document
question-answer generation and evaluation.

## Modules (`aieng.syn_data.text`)

| Module | Purpose |
|--------|---------|
| `schemas` | `QASample`, `Paragraph`, `FailureMode`, enums |
| `config` | Paths, finance document specs, defaults |
| `io` | JSON / JSONL helpers |
| `clients` | OpenAI-compatible teacher and judge clients |
| `documents` | Chunking and train/test paragraph splits |
| `datasets` | Download and load domain policy documents |
| `generation` | Zero/one/few-shot and topic-controlled Q&A generation |
| `quality` | Heuristic filtering (dedupe, leakage, length) |
| `judge` | LLM-as-judge prompts and scoring |
| `rag` | Instruction back-translation + optional lexical retrieval |
| `evaluation` | Baseline inference and score summaries |
| `pipeline` | Notebook orchestration helpers |
| `paths` | Repository root detection for notebooks |
| `small_model` | Small-model client factory |
| `sft` | LoRA dataset prep and optional training |

## Install (workspace)

```bash
uv sync --dev
```

## Quick example

```python
from aieng.syn_data.text import (
    list_domain_documents,
    load_paragraphs_from_text_files,
    chunk_text_into_paragraphs,
)

specs = list_domain_documents("finance")
paragraphs = load_paragraphs_from_text_files(specs)
print(len(paragraphs))
```

## Tests

```bash
uv run pytest aieng-synthetic-data/tests -q
```
