# SynBench

These notebooks walks through **SynBench** end to end: loading a domain, generating synthetic benchmark tasks, verifying them, and evaluating tool-calling agents.

**Prerequisites:** from the repo root, run `uv sync --dev --group synbench` to install synbench specific as well as dev dependencies, and start the first Jupyter notebook. Select the kernel and run the cells.

You can also activate the environment in terminal using `source .venv/bin/activate` command.

Then, copy env defaults into this directory and set your API key:
```bash
# from implementations/agent_benchmark_generation/
cp implementations/agent_benchmark_generation/.env.example .env   # then set OPENAI_API_KEY (and adjust models if needed)
```
---

## What is SynBench?

SynBench builds **τ-bench–style** benchmarks for customer-service agents that call tools. You define a miniature world (database + tools + policy), generate tasks with oracle solutions, verify those tasks automatically, then score agents on **outcomes** (final database state + required phrases in replies), not exact tool sequences.

### Features

- τ-inspired domain bundles (policy, tools, DB, FSM, user simulator, seed tasks)
- OpenAI-compatible API for task generation with `MOCK_LLM=1` for offline CI
- Multi-turn tool-calling agent loop and multi-role pipeline (planner / executor / user_sim / critic)
- Multi-turn dialogue with different types of personas/customers.
- Provider-agnostic LLM layer via OpenAI-compatible chat completions API
- Rule-based verification: FSM, replay, policy rules
- Outcome-first scoring (`DB` + `COMMUNICATE`, matching τ-bench semantics)



# Notebooks

Run these in order. Each notebook builds on the previous one.

### `1-check_access_to_model.ipynb`

Smoke-tests your LLM credentials against the Vector Institute OpenAI-compatible proxy. Loads `.env`, sends a short chat completion, and streams the reply so you know the API key and model work before generating tasks or running agents.

### `2-generate_and_verify_tasks.ipynb`

Walks through the **benchmark creation** path on `domains/mock_retail`:

1. Load and validate the domain bundle (policy, DB, tools, FSM, seed tasks)
2. Inspect a seed task’s oracle actions and communicate criteria
3. Replay tools in the Environment and compute a target DB hash
4. Sample constraints and generate synthetic task drafts via the LLM
5. Run the verification gate (policy rules → FSM → replay) and write passing tasks to `data/benchmarks/mock_retail/tasks.json`
6. Optionally reload and re-verify saved tasks

### `3-single_agent_evaluation.ipynb`

Evaluates a **single tool-calling agent** on the verified tasks:

1. Reload the domain and re-verify tasks from the previous notebook
2. Initialize the shared LLM client (`MOCK_LLM` or live)
3. Optionally step through one utterance of `ToolCallingLoop` (prompt → tool calls → env dispatch)
4. Run `SingleToolAgent` multi-turn dialogue (user simulator + tool loop) and score with `score_trajectory` (DB hash + communicate phrases)
5. Batch-score generated tasks with `MetricsCollector` (pass@1 and mean rewards)

### `4-multi_agent_pipeline_evaluation.ipynb`

Same setup and scoring as notebook 3, but runs **`AgentPipeline`** instead of a single agent. Per dialogue turn the roles are `user_sim` → `planner` → `executor` → `critic` (only the executor calls tools). Ends with batch metrics over the verified task set.

## Pipeline steps (overview)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD DOMAIN     policy + db.json + tools + seeds             │
├─────────────────────────────────────────────────────────────────┤
│ 2. GENERATE TASKS  LLM proposes scenario + oracle tool trace   │
├─────────────────────────────────────────────────────────────────┤
│ 3. VERIFY          policy rules → FSM → replay → target hash   │
├─────────────────────────────────────────────────────────────────┤
│ 4. RUN AGENT       multi-turn tool loop (single or pipeline)   │
├─────────────────────────────────────────────────────────────────┤
│ 5. SCORE           DB hash match + communicate_info substrings │
└─────────────────────────────────────────────────────────────────┘

```

## Evaluation semantics

`evaluation_criteria.actions` is a **reference oracle** replayed to derive the target DB hash. Agents are scored on **outcomes** (`DB`, `COMMUNICATE` by default), not exact action sequences—aligned with τ-bench.


## Adding a new domain

Copy `domains/mock_retail/` and provide:

1. `policy.md` — agent rules
2. `db.json` — initial state
3. `tools.py` — `get_tool_specs()` + `ToolKit` class
4. `state_machine.yaml` — `task_types` with `path`, `allow_write`
5. `user_simulator.yaml` — personas and goal templates
6. `tasks.seed.json` — 2–3 hand-verified seed tasks
7. `verify.py` - domain-specific rules to verify the generated tasks with.


### Domain bundle files (`domains/mock_retail/`)

- **`policy.md`** — rules the agent must follow (e.g. only cancel pending orders)
- **`db.json`** — initial world state (users, orders)
- **`tools.py`** — `get_tool_specs()` + `ToolKit` class implementing tools
- **`state_machine.yaml`** — allowed tool patterns per `task_type` (`lookup`, `mutate`)
- **`user_simulator.yaml`** — persona templates for the user simulator role
- **`tasks.seed.json`** — hand-written example tasks the LLM imitates

### Task structure

Each **Task** contains:

- **`user_scenario`** — persona, instructions, `initial_message`
- **`evaluation_criteria.actions`** — oracle tool trace (reference solution)
- **`evaluation_criteria.communicate_info`** — phrases the agent must say
- **`evaluation_criteria.reward_basis`** — typically `["DB", "COMMUNICATE"]`

### Agent modes

- **`SingleToolAgent`** — one LLM runs a multi-turn tool-calling loop
- **`AgentPipeline`** — per dialogue turn: `user_sim` → `planner` → `executor` → `critic`

### Mock vs live LLM

- **`MOCK_LLM=1`** (default in this notebook) — scripted responses, no API key, works offline
- **`MOCK_LLM=0` + `GEMINI_API_KEY`** — real Google Gemini or any other model API as long as it is OpenAI API compatible.
