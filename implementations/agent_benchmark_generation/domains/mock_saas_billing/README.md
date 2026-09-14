# Mock SaaS Billing

A larger SynBench domain for Helio Cloud billing support. It contains 48
accounts, 96 subscriptions, and 192 invoices, including nested invoice line
items. Invoices are the primary collection sampled during task generation.

## Task types

- `inquiry`
- `void_invoice`
- `refuse_void`
- `update_billing_email`
- `update_seats`

`tasks.seed.json` contains two hand-verified examples for every task type.

## Regenerating the database

From the repository root:

```bash
python implementations/agent_benchmark_generation/domains/mock_saas_billing/build_db.py
```

The builder is deterministic and preserves the pinned entities used by the
seed tasks. Commit `build_db.py` and its generated `db.json` together.

## Running the benchmark

Open `implementations/agent_benchmark_generation/5-saas-billing-scale.ipynb`.
It generates tasks stratified by task type and personality, verifies and saves
them, then reloads them for multi-agent pipeline evaluation.
