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

---

## Codebase components (map)

| Component | Location | Role |
|-----------|----------|------|
| **Domain bundle** | `domains/mock_retail/` | Policy, DB, tools, FSM, seed tasks — the simulated business |
| **Schemas** | `src/synbench/schemas/` | `Task`, `Action`, `EvaluationCriteria`, `ToolSpec` — data contracts |
| **Domain loader** | `src/synbench/domain/loader.py` | `load_domain()`, `validate_domain()` |
| **Environment** | `src/synbench/environment/` | `Environment`, `replay_actions()`, `db_hash()` — simulate tool calls |
| **FSM validator** | `src/synbench/fsm/` | Ensures oracle actions follow allowed tool patterns per `task_type` |
| **Generation** | `src/synbench/generation/` | LLM + sampler → `Task` candidates |
| **Verification** | `src/synbench/verification/` | Policy + FSM + replay → verified `Task` + `target_db_hash` |
| **LLM layer** | `src/synbench/llm/` | Provider-agnostic `LiteLLMClient` / `MockLLMClient` (default: Gemini) |
| **Agents** | `src/synbench/agents/` | `ToolCallingLoop`, `SingleToolAgent`, `AgentPipeline` (multi-role) |
| **Evaluation** | `src/synbench/evaluation/` | `score_trajectory()`, `MetricsCollector` |

---

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

The sections below follow these steps in order, with code and explanations.
```

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
