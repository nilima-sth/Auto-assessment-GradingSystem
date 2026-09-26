# Week 17 Phase 1: MLflow Baseline

## Scope

Phase 1 adds the experiment-tracking foundation only. It does not alter the Week 16
grading loop, add Evidently or Airflow, or attempt live inference without a provider.

## Implementation

- `backend/app/mlops/configs/v1.json` identifies the baseline configuration.
- `backend/app/mlops/prompts/grading_system_v1.txt` is a checked copy of the current
  Week 16 grading system prompt. The runner fails if V1 drifts from
  `backend.app.llm.base.SYSTEM_INSTRUCTION`.
- `backend/app/mlops/traces.py` serializes only existing explicit `AgentState` and
  `ToolCallRecord` fields: actions, structured arguments, summaries, errors,
  token usage, terminal state, and final grade. It does not infer or record hidden
  model reasoning.
- `run_evaluation_with_states` is an additive evaluation-harness API. It reuses the
  existing deterministic cases and returns their authoritative agent states for
  artifact generation; the original `run_case` and `run_evaluation` APIs remain
  unchanged.
- `backend/app/mlops/experiments.py` runs the harness and records parameters,
  measured metrics, and artifacts in MLflow.

Use the reproducible entry point:

```bash
uv run python -m backend.app.mlops.experiments --config v1
```

## Local tracking storage

MLflow 3.16.1 disables its legacy file-backed tracking store by default. Local
metadata therefore uses the ignored `mlflow.db` SQLite database. Artifacts use the
ignored `mlruns/` directory. This is the supported local alternative to a filesystem
tracking backend.

Open the UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## V1 baseline result

The baseline executed the six existing scripted Week 16 cases. It uses the actual
agent loop but does not contact Gemini, Ollama, or vLLM.

| Classification | Result |
| --- | --- |
| Harness and trace capture | PASS — 6 traces generated from `AgentState.trajectory`. |
| MLflow run, parameters, metrics, artifacts | PASS — run `ef47fd8a2ad34fabb3fb952eb2aa9efc`. |
| MLflow UI | PASS — started locally on port 5000; the UI API returned the experiment and V1 run. |
| Live-provider grading | BLOCKED BY LIVE PROVIDER — no Gemini key, Ollama model, or vLLM endpoint was available. |

Measured deterministic-harness results:

- Task completion rate: `1.0` (6/6)
- Tool-call correctness: `1.0` (16/16)
- Average trajectory length / agent iterations: `2.67`
- Soft failures: `1` (the intentionally injected retrieval timeout)
- Hard failures: `0`
- Provider token counts: unavailable (`0` cases with provider-reported usage)

The MLflow run stores `prompt.txt`, `config_resolved.json`,
`evaluation_summary.json`, `evaluation_records.json`, and `agent_traces.json`.

V1 is the only configuration created and run in this phase, by design. The versioned
config/prompt mechanism is ready for the required V2 and V3 experiments in a later
phase; their prompts/configurations have not been invented or evaluated here.
