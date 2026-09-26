# Week 16 Evaluation Results

I ran `backend.app.evaluation.harness` with scripted planners and tools. It still goes through `run_question_agent`, but it does not make an external LLM request. The run can be repeated locally. Its 100% result shows that the loop and safety checks behave as expected; it does not show that a live model would score 100%.

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