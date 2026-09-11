# Berka multi-table preprocessing

Prepares all eight PKDD'99 Berka tables for ClavaDDPM pipeline.

## Outputs

Written under `--output-dir`:

| File | Role |
|------|------|
| `{table}.csv` | Full table, **including** primary and foreign keys |
| `{table}_domain.json` | Feature column types only (`discrete` / `continuous`). No `size`. IDs omitted |
| `dataset_meta.json` | Parent/child graph and `relation_order` |

Optional artifacts (`--no-artifacts` to skip): `preprocess_meta.json`, `label_encoders.pkl`.

`load_tables` drops any CSV column whose name contains `_id` before applying the domain file. Keep keys in the CSVs so records can be linked; do not list them in domain JSON.

Shared rules:

- Ordinal-encode categoricals with `LabelEncoder` (sorted labels → `0..K-1`).
- Convert date columns to **days since the earliest date in that table** (optional `--date-epoch` pins a shared YYMMDD).
- IDs are **not** dropped.

## Per-table transforms

| Table | IDs (in CSV, not in domain) | Feature transforms | Discrete | Continuous |
|-------|-----------------------------|--------------------|----------|------------|
| `account` | `account_id`, `district_id` | `date` → `account_date` (day offsets) | `frequency` | `account_date` |
| `card` | `card_id`, `disp_id` | `type` → `card_type`; `issued` (strip time if present) → day offsets | `card_type` | `issued` |
| `client` | `client_id`, `district_id` | Split `birth_number` (YYMMDD; month+50 = female) into `year`, `month`, `client_date` (day), `gender` | `year`, `month`, `client_date`, `gender` | — |
| `disp` | `disp_id`, `client_id`, `account_id` | `type` → `disp_type` | `disp_type` | — |
| `district` | `A1` renamed to `district_id` | Coerce `A4`–`A16` to numeric; `?` filled with 0 | `A2`, `A3` | `A4`–`A16` |
| `loan` | `loan_id`, `account_id` | `date` → `loan_date` | `status` | `loan_date`, `amount`, `duration`, `payments` |
| `order` | `order_id`, `account_id` | Fill missing `k_symbol` / `bank_to` with `""` (a lone space is its own level) | `bank_to`, `k_symbol` | `account_to`, `amount` |
| `trans` | `trans_id`, `account_id` | `date` → `trans_date`, `type` → `trans_type`; fill cats; partner `account` empty → 0 | `trans_type`, `operation`, `k_symbol`, `bank` | `trans_date`, `amount`, `balance`, `account` |

CSV column order is IDs first, then features in the same order as the domain files.

**`district_id`:** the district key must be named `district_id`, not `A1`. Otherwise `load_tables` would treat `A1` as a modeled feature.


## Difference from single-table Berka

Single-table preprocessing drops `trans_id` / `account_id` and writes train/holdout splits. This pipeline keeps all relational IDs and writes one CSV per table. Both pipelines write type-only domain files (`discrete` / `continuous`; no `size`).
