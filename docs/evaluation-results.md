# Week 16 Evaluation Results

Generated from `backend.app.evaluation.harness` using the actual `run_question_agent` loop and deterministic scripted planners/tools. The scripted run is reproducible and does not call an external LLM. Its 100% result validates loop mechanics and safety behavior only; it is not evidence that a live LLM planner would achieve 100%.

| Case | Outcome | Correct Tools | Steps | Input Tokens | Output Tokens | Total Tokens | Failure |
| ---- | ------- | ------------- | ----- | ------------ | ------------- | ------------ | ------- |
| direct-grade | completed | 2/2 | 2 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | none |
| rag-needed | completed | 3/3 | 3 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | none |
| rewrite-retrieval | completed | 4/4 | 4 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | none |
| verification-replan | completed | 5/5 | 5 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | none |
| ambiguous-review | manual_review | 1/1 | 1 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | none |
| retrieval-timeout | manual_review | 1/1 | 1 | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | 0 (deterministic simulation; no LLM request) | Soft failure: injected retrieval timeout |

## Aggregate

- Cases: 6
- Task completion rate: 6/6 = 100%
- Tool-call correctness: 16/16 = 100%
- Average trajectory length: 2.67 actions
- Provider-reported LLM tokens: 0 for all 6 cases because every case was a deterministic simulation with no LLM request
- Hard failures: 0
- Soft failures: 1
- Cascading soft failures: 0

## Failure Classification

The retrieval timeout is a soft failure because the retrieval tool failed, the failure was recorded, no evidence was invented, no grade was finalized, and the agent returned manual review. It is not cascading because the failed result was not passed into grading or finish actions.

The harness accepts injected real planners/providers through `run_evaluation(..., planner_factory=..., grade_fn_factory=..., retrieve_fn=...)`. No live run was recorded: Gemini credentials were absent, the vLLM endpoint was unreachable, and Ollama had no locally installed model. Therefore the deterministic metrics must not be presented as live-agent success metrics.
