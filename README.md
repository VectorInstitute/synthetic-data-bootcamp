# Multi-modal Synthetic Data Generation Bootcamp

This repository contains the reference implementations created by the Vector AI
Engineering team for the Synthetic Data Generation Bootcamp. It provides reference materials and implementations to help teams explore and address modern AI data bottlenecks by leveraging synthetic data across different modalities.

## Bootcamp Topics

This bootcamp has four main themes. In each team we try to target a known data bottleneck of modern AI systems, and show how synthetic data can be leveraged to target these bottlenecks.


## Repository Structure

- **implementations/**: Implementations are organized by topics. Each topic has its own directory containing notebooks, and a README for guidance. 
- **pyproject.toml**: The `pyproject.toml` file in this repository configures various build system requirements and dependencies, centralizing project settings in a standardized format.
- **aieng-synthetic-data/**: While the notebooks and codes under `implementations/**` directory are kept lightweight for easier navigation and learning purposes, they are backed by a small core library which lives under `aieng-synthetic-data/**`. This directory acts as an independent library and can be edited to change the behavior of reference implementations or to extend them. It is installed as a dependency in this repo's environments.

### Implementations Directory

Each topic within this bootcamp has a dedicated directory in the `implementations/` directory. In each directory, there is a README file that provides an overview of the topic, prerequisites, and notebook descriptions.


| Topic | Learning Topics | 
| -------- | -------- | 
| QA Generation for SLM Alignment  | Generate QAs from an unstructured source data using a strong LLM, Explore grounded generation with prompting techniques and instruction back translation (increase fidelity) and Topic controlled generation (increase diversity), SFT a local SLM and evaluate it across known failure modes, Explore targeted data generation for Direct Preference Optimization (DPO)  | 
| Agent Benchmark Generation   | Design and explore an adaptive framework to synthesize benchmarks for multi-step customer service agents. Explore existing ideas and extend it to your domain. Evaluate single and multi-agent systems. Control fidelity by constrained generation and increase diversity across user personas. Define a new domain with a new database, tools, and policies. Inspired by the leading customer service benchmark τ-bench |
| Tabular Data Synthesis  | Generate single table and relational multi-table datasets while preserving relational statistics and preserve the database schema, evaluate quality and  |
| Row 2 A  | Row 2 B  |

Topic | Description
- QA Generation for SLM Alignment | Generate QAs from an unstructured source data using a strong LLM, Explore grounded generation with prompting techniques and instruction back translation, SFT a SLM, Explore targeted data generation for Direct Preference Optimization (DPO)
- Agent Benchmark Generation |
- Tabular Data Synthesis |
- Edge Case Image Synthesis | 

## Getting Started

To get started with this bootcamp (*Change or modify the following steps based your needs.*):
1. Clone this repository to your machine.
2. *Include setup and installation instructions here. For additional documentation, refer to the `docs/` directory.*
3. Begin with each topic in the `implementations/` directory, as guided by the README files.

## License
*Add appropriate LICENSE for this bootcamp in the main directory.*
This project is licensed under the terms of the [LICENSE](LICENSE.md) file located in the root directory of this repository.

## Contribution
*Add appropriate CONTRIBUTING.md for this bootcamp in the main directory.*
To get started with contributing to our project, please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## Contact Information

For more information or help with navigating this repository, please contact Fatemeh Tavakoli at [fatemeh.tavakoli@vectorinstitute.ai](fatemeh.tavakoli@vectorinstitute.ai) .
