# Data

## `cardio_train.csv`

- **Source:** [Kaggle · Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (sulianova). Kaggle lists the license as "Unknown". The dataset is on Vector's pre-approved list for this bootcamp.
- **Size:** 70,000 patients × 13 columns. Semicolon-separated, no missing values.
- **Columns:**
  - `id`
  - `age` (in days)
  - `gender` (1 / 2)
  - `height` (cm) and `weight` (kg)
  - `ap_hi` / `ap_lo`: systolic / diastolic blood pressure
  - `cholesterol` and `gluc`: 1 = normal, 2 = above normal, 3 = well above normal
  - `smoke`, `alco`, `active`: 0 / 1
  - `cardio`: target, 1 = has cardiovascular disease
- **Known issues:**
  - Blood-pressure entry errors (ap_hi ranges from −150 to 16020; ap_lo from −70 to 11000).
  - 24 duplicate rows.
  - About 2.1% of rows fail basic medical-range rules and are removed in preprocessing.

Treat this file as if it were private patient data. Only the train split is used to fit the generator, and only synthetic data should leave the project.
