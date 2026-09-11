# Multi-modal Synthetic Data Generation Bootcamp

This repository contains reference implementations from the Vector AI Engineering team for the Synthetic Data Generation Bootcamp. Use these materials to explore how synthetic data can address data bottlenecks in modern AI systems across text, agents, tabular data, and images.

## Bootcamp Topics

This bootcamp has four main themes. In each one we try to target a known data bottleneck of modern AI systems, and show how synthetic data can be leveraged to target these bottlenecks.

1. **QA generation for SLM alignment** — generate grounded question–answer data to fine-tune and align small language models.
2. **Agent benchmark generation** — synthesize evaluation tasks for multi-step customer service agents.
3. **Tabular data synthesis** — generate single-table and relational multi-table datasets while preserving schema and statistics.
4. **Edge case image synthesis** — generate rare or hard-to-collect visual scenarios for downstream vision tasks.

## Repository Structure

- **implementations/**: Materials organized by topic. Each topic directory includes notebooks and a README with guidance.
- **pyproject.toml**: Project settings and dependency groups for each topic.
- **aieng-synthetic-data/**: A small core library that backs the notebooks. The notebooks under `implementations/` stay lightweight for learning; this package is installed into the project environments. You can edit it to change or extend the reference implementations.

### Implementations Directory

Each topic has its own directory under `implementations/`. The README in that directory covers the topic overview, prerequisites, and notebook descriptions.

| Topic | Learning topics |
| -------- | -------- |
| [QA Generation for SLM Alignment](implementations/qa_text_generation) | Generate Q&A pairs from unstructured source data with a strong LLM. Explore grounded generation with prompting techniques and instruction back-translation (to increase fidelity) and topic-controlled generation (to increase diversity). Supervised fine-tune a local SLM and evaluate it on known failure modes. Explore targeted data generation for Direct Preference Optimization (DPO). |
| [Agent Benchmark Generation](implementations/agent_benchmark_generation) | Design and explore an adaptive framework for synthesizing benchmarks for multi-step customer service agents. Build on existing ideas and extend them to your domain. Evaluate single-agent and multi-agent systems. Control fidelity with constrained generation and increase diversity across user personas. Define a new domain with a new database, tools, and policies. Inspired by the leading customer service benchmark [τ-bench](https://taubench.com/). |
| [Tabular Data Synthesis](implementations/tabular_data) | Generate single-table and relational multi-table datasets while preserving relational statistics and the database schema. Evaluate the quality and privacy of the synthesized data. |
| [Edge Case Image Synthesis](implementations/edge_case_image_generation) | Generate visual scenarios that are rare, difficult, expensive, or impractical to collect in sufficient quantities from the real world. Compare generation approaches, including different edit methods (diffusion + ControlNet vs. diffusion inpaint vs. a general edit model vs. VLMs). Measure quality with a vision-language model (VLM) in the loop, and measure utility by the downstream task performance boost. |

## Getting Started

1. Clone this repository:

```bash
git clone <repo-url>
cd synthetic-data-bootcamp
```

2. Open a topic directory under `implementations/` and follow its README.
3. From the repository root, install `uv` (if needed) and sync the dependency group for that topic.

Install uv if you have not already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

All dependency groups are defined in the root `pyproject.toml`. Install only the group you need **from the repository root**:

| Topic | Group name | Install command |
| :---     | ---    | ---     |
| [QA Generation for SLM Alignment](implementations/qa_text_generation) | `text-sft` | `uv sync --dev --group text-sft` |
| [Agent Benchmark Generation](implementations/agent_benchmark_generation) | `synbench` | `uv sync --dev --group synbench` |
| [Tabular Data Synthesis](implementations/tabular_data) | `tabular-data` | `uv sync --dev --group tabular-data` |
| [Edge Case Image Synthesis](implementations/edge_case_image_generation) | `edge-case-image-generation` | `uv sync --dev --group edge-case-image-generation` |

### API keys

To be added.

## License
This project is licensed under the terms of the [LICENSE](LICENSE.md) file located in the root directory of this repository.

## Contribution
To get started with contributing to our project, please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## Contact

For more information or help with navigating this repository, please contact Fatemeh Tavakoli at [fatemeh.tavakoli@vectorinstitute.ai](fatemeh.tavakoli@vectorinstitute.ai) .
