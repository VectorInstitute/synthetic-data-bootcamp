## TabDDPM: modelling tabular data with diffusion models

TabDDPM (Tabular Denoising Diffusion Probabilistic Model) is a generative framework that applies diffusion processes to tabular data by combining continuous Gaussian noise for numerical variables with discrete multinomial noise for categorical variables. It is widely considered a state-of-the-art (SOTA) tabular generative model.

### Layout

```
single_table/
├── data_processing/      # Download trans, preprocess, train/holdout split
├── training/             # TabDDPM training pipeline
├── synthesizing/         # Synthesize tabular data with the trained model
├── data/                 # Raw/preprocessed input data (gitignored)
├── results/              # Model checkpoints and synthesis outputs (gitignored)
```
#### Suggested path

**Single table:** [`single_table/README.md`](single_table/README.md) → `data_processing/data_processing.ipynb` → `training/tabddpm_training.ipynb` → `synthesizing/tabddpm_synthesizing.ipynb` → [`evaluation/`](evaluation/) quality and privacy notebooks.

#### Install and activate the virtual env
From the repo root, run `uv sync --dev --group tabular-data` to install tabular data implementation as well as dev dependencies, and start the first Jupyter notebook. Select the kernel and run the cells.


### Background: Diffusion models
Every diffusion model (image, audio, or tabular) is built on a simple framework: take real data, destroy it with noise in a slow, controlled way, then train a network to undo that destruction one small step at a time.
The same recipe works on a row of a spreadsheet instead of a grid of pixels in an image. We just need to define what "adding noise" and "removing noise" mean for spreadsheet-shaped data.

#### The difference:
A spreadsheet row is not one homogeneous thing. A pixel is always a continuous brightness value, and neighboring pixels are spatially related. A table row is a grab-bag: `age` is a continuous number, `city` is a category with no natural ordering, `has_subscription` is binary. We can't just "add Gaussian noise" to a category the way you can to a number (pixel in an image)

### TabDDPM

They combine gaussian diffusion and multinomial diffusion to model numerical and categorical features, respectively. Gaussian diffusion models operate in continuous spaces where forward and reverse processes are characterized by Gaussian distributions, while multinomial diffusion are designed to generate categorical data and employs categorical distribution.


<div align="center">
  <img src="./images/tabddpm.png" alt="TabDDPM" width="600" height="280">
</div>

#### Training
You never need to actually run the forward process step-by-step during training. Because each forward step just adds a small amount of Gaussian (or multinomial) noise, the closed-form noise level at any timestep `t` can be computed directly in one shot:


Numeric columns:


$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0
+ \sqrt{1 - \bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$


Categorical columns (one-hot encoded, K possible categories): gets blended into uniform distribution.



$$
x_t = \bar{\alpha}_t x_0 + \frac{1 - \bar{\alpha}_t}{K}\mathbf{1}
$$


Alpha is a precomputed number between 0 and 1 that shrinks as t grows (close to 1 at t=0, close to 0 at t=T). It determines how much noise should be added to get from x_0 to x_t.

**Loss**: They compare the predicted noise to the ground truth added noise: MSE (numeric) + KL (categorical)

## TabDDPM compared to LLMs
Compared to using general-purpose Large Language Models (LLMs) out of the box, TabDDPM is significantly superior for pure tabular synthesis for three key reasons:

- Native Heterogeneous Modeling: Out-of-the-box LLMs process text sequentially and often convert numerical tables into string formats (e.g., CSV or JSON prompts), which forces the model to learn arithmetic rules and data schemas implicitly rather than directly modeling joint probability distributions.

  - Autoregressive probability modelling(LLMs): $P(T_1, T_2, \dots, T_N) = \prod_{i=1}^N P(T_i \mid T_1, \dots, T_{i-1})$
  - Modelling join probability density function(Diffusion) : $P(X_1, X_2, \dots, X_d)$

- Numerical Precision & Joint Probability: LLMs treat numbers as text tokens, making them prone to floating-point drift, invalid formats, and hallucinated boundaries; TabDDPM directly optimizes continuous distributions in feature space.

- Privacy Control & Compute Efficiency: TabDDPM is lightweight, fast to fine-tune on private tabular schemas, and straightforward to train with Differential Privacy guarantees (DP-SGD), whereas enforcing mathematical privacy bounds on massive LLMs without destroying model utility remains a major hurdle.


Sources:
- https://research.yandex.com/blog/tabddpm-modelling-tabular-data-with-diffusion-models.
- TabDDPM official repo: https://github.com/yandex-research/tab-ddpm.
- Tabddpm Paper: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf.

Additional reading: https://www.emergentmind.com/topics/tabular-denoising-diffusion-probability-models-tabddpm

## Example dataset: Berka
The Berka dataset is a collection of financial information from a Czech bank. The dataset deals with over 5,300 bank clients with approximately 1,000,000 transactions. Additionally, the bank represented in the dataset has extended close to 700 loans and issued nearly 900 credit cards, all of which are represented in the data."
Download link: https://www.kaggle.com/datasets/marceloventura/the-berka-dataset

In the single-table reference implementation, we work with the transaction table of this dataset.
- `trans.csv` table information: https://webpages.charlotte.edu/mirsad/itcs6265/group1/transaction_domain.html

The whole dataset generation is covered in the `multi_table` reference implementation.



## Choose the Right Generative Approach

Tabular data is notoriously tricky because it mixes continuous numbers, discrete categories, missing values, and skewed distributions.

| Model Class              | Top Algorithms                             | Best Used For                                               | Key Strengths                                                            | Limitations                                                    |
|--------------------------|--------------------------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------|
| Tabular Diffusion        | TabDDPM                                    | High-accuracy ML modeling & complex tabular relations       | Highest statistical fidelity and handles mixed types exceptionally well   | Slower training and inference than VAEs/Copulas                |
| Conditional Deep Learning| CTGAN, TVAE                                | Standard enterprise datasets with heavily skewed categories  | Handles multimodal numeric columns and highly imbalanced categories well  | GANs can suffer from training instability/mode collapse         |
| Statistical / Copula     | Gaussian Copula, Synthpop                  | Quick baselines, small datasets, light analytics            | Fast to run, lightweight, highly interpretable                           | Misses non-linear, complex multi-column interactions           |
| LLM / Autoregressive     | GReaT (Generation of Realistic Tabular Data)| Datasets with embedded free text or rich schema context      | Seamlessly blends structured tables with unstructured text fields         | High compute footprint; requires careful prompt engineering     |
