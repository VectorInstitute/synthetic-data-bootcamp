# Tabular data reference implementations

Two diffusion-based pipelines on the [Berka](https://www.kaggle.com/datasets/marceloventura/the-berka-dataset) Czech bank dataset are implemented. The single-table pipeline trains a tabular diffusion model on the transaction table of this dataset, and the multi-table implementation, trains a diffusion model on the whole relational dataset. Both pipelines cover training, synthesizing, and evaluation steps.

| Track | Model | What it covers |
|-------|--------|----------------|
| [`single_table/`](single_table/) | [TabDDPM](https://research.yandex.com/blog/tabddpm-modelling-tabular-data-with-diffusion-models) | One table (`trans`): mixed Gaussian + multinomial diffusion |
| [`multi_table/`](multi_table/) | [ClavaDDPM](https://github.com/VectorInstitute/midst-toolkit) | All eight tables: cluster parent–child groups, train one model per edge, synthesize root-first |

Both use [midst-toolkit](https://github.com/VectorInstitute/midst-toolkit) and [SynthEval](https://github.com/schneiderkamplab/syntheval) toolkits.

## To Run:
From the repo root, run `uv sync --dev --group tabular-data` to install tabular data implementation as well as dev dependencies, and start the first Jupyter notebook. Select the kernel and run the cells.

### Hardware requirements

To run training notebooks, make sure you have access to a strong GPU. You can also run the code on your computer, but you need to make the model smaller and adjust training parameters such as iterations.

## Layout

```
tabular_data/
├── utils.py                          # Berka download helpers (single- and multi-table)
├── single_table/                     # TabDDPM — see README.md for model + dataset notes
│   ├── data_processing/              # Download trans, preprocess, train/holdout split
│   ├── training/                     # tabddpm_training.ipynb + config.yaml
│   ├── synthesizing/                 # tabddpm_synthesizing.ipynb + config.yaml
│   ├── data/  results/               # Raw/preprocessed data and checkpoints (gitignored)
│   └── images/
├── multi_table/
│   ├── data_preprocessing/           # Eight tables for ClavaDDPM — README.md
│   ├── training/                     # Clustering + per-edge training — README.md
│   ├── synthesizing/                 # Root-first sampling + multi-parent matching — README.md
│   ├── evaluation/                   # Relational quality (e.g. account → loan)
│   ├── data/berka/                   # Preprocessed CSVs, *_domain.json, dataset_meta.json
│   └── images/
└── evaluation/                       # Column-level quality and privacy (single-table style)
    ├── quality/  privacy/            # Notebooks (incl. membership inference)
    └── preprocessing.py
```

Each `config.yaml` is loaded with Hydra from the notebook in that folder. Training configs are small by default (few timesteps/iterations) so a first run is cheap; scale them up for a real model.

## Suggested path

**Single table:** [`single_table/README.md`](single_table/README.md) → `data_processing/data_processing.ipynb` → `training/tabddpm_training.ipynb` → `synthesizing/tabddpm_synthesizing.ipynb` → [`evaluation/`](evaluation/) quality and privacy notebooks.

**Multi table:** [`data_preprocessing/README.md`](multi_table/data_preprocessing/README.md) (raw files via `download_and_save_multi_table_data`, then `pre_process_berka_all_tabels.py` or the notebook) → [`training/README.md`](multi_table/training/README.md) / `ClavaDDPM_training.ipynb` → [`synthesizing/README.md`](multi_table/synthesizing/README.md) / `ClavaDDPM_synthesizing.ipynb` → `multi_table/evaluation/multi_table_quality.ipynb` for 1-hop relational metrics. Use [`evaluation/`](evaluation/) for per-table column metrics on a generated table.

