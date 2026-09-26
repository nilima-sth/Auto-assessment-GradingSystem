# Week 17 Phase 4 — final Track B audit and submission preparation

Audit date: 2026-09-26. This document audits the implementation and recorded
artifacts; it adds no new MLOps behavior, prompt version, or orchestration system.

## Requirement compliance matrix

| Track B requirement | Status | Concrete evidence |
| --- | --- | --- |
| uv environment | PASS | Root `pyproject.toml`; `uv sync` and a clean-copy `uv sync --frozen` completed. |
| `pyproject.toml` | PASS | Root project manifest is present and resolves 245 packages. |
| `uv.lock` | PASS | Root lockfile; `uv lock --check` passed. |
| One-command setup | PASS | `uv sync` is documented and succeeds from a clean source copy. |
| Explicit prompt/config versions | PASS | `backend/app/mlops/configs/v1.json` through `v3.json` and matching prompt files. |
| At least three versions | PASS | V1, V2, and V3 recorded in Phase 2 and regression runs. |
| MLflow tracking | PASS | `backend/app/mlops/tracking.py`, SQLite `mlflow.db`, and recorded runs. |
| Meaningful parameters | PASS | `resolved_parameters` logs provider/model, prompt IDs, LLM controls, RAG controls, and iteration limit. |
| Meaningful metrics | PASS | Phase 2 records completion, tool correctness, iterations, latency, errors, and failure classes. |
| Full traces | PASS | `agent_traces.json` is logged by `experiments.py` from `AgentState.trajectory`. |
| Tool calls and arguments | PASS | `traces.py` serializes each action and structured `arguments`. |
| Tool results | PASS | `traces.py` serializes `result_summary`, success, and error. |
| Iterations | PASS | Trace has per-step `iteration` and `total_iterations`. |
| Termination reason | PASS | Trace has `status` and `termination_reason`. |
| Representative traces | PASS | `select_representative_traces` writes `representative_traces.json`. |
| Failure-driven V1 → V2 | PASS | [Phase 2 notes](week17-phase2-experiments.md) link invalid/repeated verification traces to planner state guards. |
| Failure-driven V2 → V3 | PASS | [Phase 2 notes](week17-phase2-experiments.md) link exact-match under-scoring to grading calibration. |
| Fixed regression set | PASS | Six fixed cases in `backend/app/mlops/golden/regression_v1.json`. |
| Approved golden/reference answers | PASS | Each golden case has a source, expected status/marks/actions, and `reference_response`; methodology is documented in [Phase 3](week17-phase3-regression.md). |
| Evidently LLM evaluation | PASS | `regression.py` uses Evidently 0.7.23 `LLMEval` and `BinaryClassificationPromptTemplate`. |
| At least two judge checks | PASS | `reference_correctness` and `workflow_adherence`, with category and reasoning retained per case. |
| `pct_tests_passed` in MLflow | PASS | Fresh verified V1/V2/V3 regression runs log `pct_tests_passed`, using the same value as preserved `regression_pass_percentage`. |
| Judge sanity checking | PASS | `_project_sanity` checks status, required actions, tool-call limit, grade presence, and marks; disagreements are retained. |
| Evidently HTML reports | PASS | `reports/evidently/v1_regression.html`, `v2_regression.html`, and `v3_regression.html` exist and are not ignored. |
| README documentation | PASS | Root README documents setup, app use, Ollama, MLflow, experiment/regression commands, results, limits, and Airflow status. |
| Optional Airflow | OPTIONAL / NOT ATTEMPTED | Explicitly optional bonus work; not installed or implemented. |

No mandatory Track B requirement is missing in this final audit. `regression_pass_percentage` remains for backward compatibility, while `pct_tests_passed` is the assignment-facing metric name.

## Environment and reproducibility verification

| Check | Result |
| --- | --- |
| `uv lock --check` | PASS |
| `uv sync` | PASS |
| `uv run python -m compileall backend frontend tests` | PASS |
| `uv run pytest -q` | PASS — 44 passed |
| Clean temporary source copy | PASS — excluded `.venv`, local `.env`, databases, MLflow data, and Chroma; `uv sync --frozen` then 44 tests passed. The temporary copy was moved to system trash after verification. |
| FastAPI | PASS — started on a temporary local port; `GET /api/health` returned HTTP 200 and `{"status":"ok"}`. |
| MLflow | PASS — local `mlflow ui` started on a temporary local port; `/health` returned HTTP 200. |
| Experiment CLI | PASS — `python -m backend.app.mlops.experiments --help` exposes config, tracking URI, experiment name, and deterministic/live modes. |
| Regression CLI | PASS — `python -m backend.app.mlops.regression --help` exposes `v1`, `v2`, `v3`, and `all`. |
| Ollama model | Previously verified live with local `qwen2.5:7b`; it was not rerun in this final audit to avoid unnecessary expensive calls. |

To reproduce live calls, first run `ollama pull qwen2.5:7b`, then use the commands in the README. Actual live reproducibility still depends on local Ollama service/model availability and suitable hardware.

## Experiment evidence

Phase 2 live MLflow runs:

| Version | Run ID | Completion | Tool correctness | Avg. iterations | Latency | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V1 | `2a00595ef67642cb93e63ad01a59f0ea` | 0.333 | 0.941 | 2.83 | 40.85 s | 2 |
| V2 | `b665af9936f446c98186c944e980dfc4` | 0.167 | 1.000 | 3.50 | 73.65 s | 2 |
| V3 | `f188ccc5a3394912abf5bb9bd0c69447` | 0.333 | 1.000 | 1.83 | 13.73 s | 0 |

V1 exposed verification-order, repeated-verification, maximum-iteration, and calibration weaknesses. V2 added planner guards and improved tool correctness, but completion worsened and latency increased: a measured regression. V3 retained V2 planning, calibrated exact matches, and removed provider/tool errors while reducing iterations and latency. It remains limited rather than generally validated.

## Regression evidence

| Version | Regression run | Passed | Percentage | Failed cases |
| --- | --- | ---: | ---: | --- |
| V1 | `4905b1f12a13446db24b65730972b3a5` | 1/6 | 16.67% | exact-match, partial, incorrect, retrieval, ambiguous OCR |
| V2 | `2be58ca9872243809efb245b45b251f1` | 1/6 | 16.67% | exact-match, partial, incorrect, retrieval, ambiguous OCR |
| V3 | `9e17f564ef0447bb8e4f93eb70e5c21f` | 2/6 | 33.33% | partial, incorrect, retrieval, ambiguous OCR |

The evidence reports are [V1](../reports/evidently/v1_regression.html),
[V2](../reports/evidently/v2_regression.html), and
[V3](../reports/evidently/v3_regression.html). V3 improves the fixed set but is
not promotable automatically: candidate/reference/judge/contract disagreements
remain in the MLflow `regression_results.json` artifacts for human review.

Fresh compliance reruns, created after adding the assignment-facing metric, verify
that both percentage keys are equal in actual MLflow records:

| Version | Fresh regression run | `pct_tests_passed` | `regression_pass_percentage` |
| --- | --- | ---: | ---: |
| V1 | `a9eba64e6b884ec8952056509f6bb4b3` | 16.666666666666668 | 16.666666666666668 |
| V2 | `4ad434109ce94b3c9dc396501f023b50` | 16.666666666666668 | 16.666666666666668 |
| V3 | `438a8c2460434881903ffe2241926a34` | 33.333333333333336 | 33.333333333333336 |

## README, hygiene, and secret audit

The README was revised for a reviewer who starts at the repository root. It
preserves the Week 15/16 application story while making Week 17 setup, commands,
versions, results, reports, caveats, and optional Airflow status explicit.

`.gitignore` excludes `.env` files, `.venv`, Python caches, SQLite runtime files,
`mlruns`, `mlartifacts`, Chroma data, models/caches, logs, temp files, and editor
swap files. It does **not** ignore `reports/`, `pyproject.toml`, `uv.lock`, source,
prompts/configs, golden data, docs, tests, or the final Evidently HTML reports.

A pattern-based scan of non-runtime source/docs (excluding the ignored local
`backend/.env` and generated report HTML) found only expected configuration
placeholders, environment-variable names, test fixtures, and documentation—not a
credential value. A masked key-only check of the ignored local `backend/.env`
found normal runtime settings and a blank `GEMINI_API_KEY`; no credential-like
key files were found. Its values were neither printed nor copied into the
clean-source test.

## Known limitations and optional work

- V3 still fails partial-credit, incorrect-answer, retrieval, and ambiguous-OCR
  cases in the fixed set.
- Local Qwen results and timings depend on local model/runtime state; Ollama CLI
  token counts are unavailable.
- The Phase 2 harness uses deterministic retrieval and has indistinguishable
  inputs with different expected tool paths.
- OCR and vLLM full runtime behavior depend on local model availability, input
  quality, GPU/VRAM, and first-use downloads.
- Airflow is **OPTIONAL BONUS — NOT IMPLEMENTED**.

## Repository and Git status

`git rev-parse --is-inside-work-tree` reports that this ZIP-derived working
directory has no Git metadata. No `git init`, commit, remote change, or push was
performed.

## Remaining manual submission steps

1. Review the three HTML reports and MLflow artifacts; do not promote V3 solely
   from its 2/6 result.
2. Choose one safe Git path: (A) copy these changed files into the original Week
   16 Git repository and commit there, or (B) after confirming no original repo is
   available, initialize this folder as the Week 17 submission repository.
3. Before any commit, include source, `pyproject.toml`, `uv.lock`, prompts/configs,
   golden data, docs, tests, machine-readable comparison, and `reports/evidently`;
   exclude the ignored runtime/secrets listed above.
4. Set a remote and push only after verifying the intended repository and branch.
