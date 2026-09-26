# Week 17 agent experiments

## Method

The existing Week 16 question-level agent loop was evaluated with three explicit
prompt/configuration versions. All live planner and grading calls used local
Ollama model `qwen2.5:7b`; harness retrieval remained deterministic. MLflow
records configuration parameters, aggregate metrics, complete safe traces, and
representative traces. Trace artifacts contain explicit actions, arguments,
result summaries, errors, iterations, termination state, and final grades; they
do not include hidden model reasoning.

V1 is the Week 16 baseline. Its traces exposed invalid verification before a
grade, repeated verification, maximum-iteration exits, and under-scoring of an
exact model-answer match. V2 adds planner state guards only. V3 retains those
guards and adds targeted exact-match grading calibration.

## Recorded live results

| Version | MLflow run | Completion | Tool correctness | Avg. iterations | Latency | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V1 | `2a00595ef67642cb93e63ad01a59f0ea` | 0.333 | 0.941 | 2.83 | 40.85 s | 2 |
| V2 | `b665af9936f446c98186c944e980dfc4` | 0.167 | 1.000 | 3.50 | 73.65 s | 2 |
| V3 | `f188ccc5a3394912abf5bb9bd0c69447` | 0.333 | 1.000 | 1.83 | 13.73 s | 0 |

V2 improved tool correctness but reduced completion and increased latency, so it
is a measured regression. V3 removed recorded provider/tool errors and reduced
iterations and latency while preserving V1 completion. This small harness is an
operational evaluation, not a claim of general grading accuracy.

Run a deterministic baseline:

```bash
uv run python -m backend.app.mlops.experiments --config v1
```

Run a live local version:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.experiments --config v3 --execution-mode ollama-live
```
