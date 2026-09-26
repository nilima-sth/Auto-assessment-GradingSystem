# Week 17 Phase 3 — fixed golden regression evaluation

## Scope

This phase adds a fixed regression suite alongside, not inside, the Week 16 agent
harness. It does not change the agent loop, introduce a V4 prompt, add Airflow,
or automate promotion.

## Evidently implementation

The project uses `evidently[llm]` 0.7.23. The current API used here is
`BinaryClassificationPromptTemplate` plus `LLMEval`, attached to an Evidently
`Dataset`, and `Report(...).save_html(...)` for the rendered report. It supports
local Ollama natively through Evidently's LiteLLM-backed `provider="ollama"`
wrapper and `OllamaOptions(api_url="http://localhost:11434")`; no cloud key or
mocked judge is used in live runs.

The judge is local `qwen2.5:7b`. Each row stores both its category and reasoning
for two independent binary checks:

1. `reference_correctness`: whether the candidate preserves the approved
   reference outcome, without contradictions, material omissions, or an
   unjustified score/status.
2. `workflow_adherence`: whether terminal status, required actions, score policy,
   and the tool-call guard agree with the approved workflow.

`UNKNOWN` is configured as `FAIL` so an uncertain judge result cannot pass a
regression.

## Fixed golden set

[`backend/app/mlops/golden/regression_v1.json`](../backend/app/mlops/golden/regression_v1.json)
contains six hand-reviewed, versioned inputs. They are grounded in the existing
Week 16 deterministic cases and the `GradeEvaluation` contract, not copied from
V3 output:

- exact answer and partial-credit indexing answers;
- an explicitly incorrect encryption claim;
- a reference-grounded retrieval case;
- ambiguous OCR requiring manual review without a grade;
- a post-grade loop guard derived from the Phase 2 repeated-action weakness.

The fixed retrieval function supplies only the case-owned evidence. This makes
the regression reproducible without mutating Chroma or relying on documents in a
developer's local vector store.

## Run and inspect

Run one configuration from the project root:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.regression --config v3
```

Run all fixed configurations:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.regression --config all
```

The command prints case count, passes, percentage, failed case IDs, HTML report
path, and MLflow run ID. It writes reports to:

- `reports/evidently/v1_regression.html`
- `reports/evidently/v2_regression.html`
- `reports/evidently/v3_regression.html`

Each MLflow run is tagged `evaluation_type=fixed_golden_regression`,
`prompt_version`, `judge_provider`, `judge_model`, and
`promotion=interpretation_only`. It logs resolved parameters, fixed input cases,
per-case candidate/reference/verdict/reasoning/sanity results, summary metrics,
and the report artifact.

## Sanity review and interpretation

The local judge is not the only acceptance signal. The runner separately checks
the project-owned contract for expected status, required actions, maximum tool
calls, grade presence, and expected marks. A case passes only when both judge
checks and this independent contract check pass. Any disagreement is retained as
`Judge/project-contract disagreement: preserve for human review; no automatic
promotion.` in the per-case artifact.

This is a release-review input, not an automated promotion rule. Review failed
cases, judge explanations, and the generated HTML before deciding whether a
prompt configuration should advance.

## Results

All runs below used the fixed six cases, the configured local `qwen2.5:7b`
agent, and the same local Qwen Evidently judge. These are separate MLflow runs
from the Phase 2 harness experiments.

| Config | Passes | Percentage | MLflow run | Report |
| --- | ---: | ---: | --- | --- |
| V1 | 1 / 6 | 16.67% | `4905b1f12a13446db24b65730972b3a5` | `reports/evidently/v1_regression.html` |
| V2 | 1 / 6 | 16.67% | `2be58ca9872243809efb245b45b251f1` | `reports/evidently/v2_regression.html` |
| V3 | 2 / 6 | 33.33% | `9e17f564ef0447bb8e4f93eb70e5c21f` | `reports/evidently/v3_regression.html` |

V1 failed `exact-match-indexing`, `partial-indexing`, `incorrect-indexing`,
`retrieval-evidence-required`, and `ambiguous-ocr-manual-review`. V2 failed the
same five cases. V3 failed `partial-indexing`, `incorrect-indexing`,
`retrieval-evidence-required`, and `ambiguous-ocr-manual-review`; it passed the
exact-match and loop-guard cases.

The sanity check found judge/project-contract disagreements that deserve manual
inspection: V1 ambiguous OCR; V2 retrieval evidence; V3 retrieval evidence and
ambiguous OCR. The candidate, approved reference, both judge explanations, and
the contract explanation are retained in each run's `regression_results.json`
artifact. The low pass rates mean none of these versions should be promoted from
this evidence alone; V3 is an improvement in this fixed set, not a release gate.
