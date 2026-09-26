# Week 16 Checklist

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Genuine observation-action loop | `backend/app/agent/loop.py:run_question_agent` | Planner is called after every state update; tests cover multiple trajectories. |
| Non-fixed action ordering | `AgentDecision` and planner provider methods | `tests/test_agent_loop.py` covers direct grading, repeated retrieval, verification replanning, and clarification. |
| Multiple iterations and stopping | `AGENT_MAX_ITERATIONS`, loop terminal statuses | `tests/test_agent_loop.py::test_max_iterations_does_not_return_confident_grade`. |
| Context engineering | `backend/app/agent/context.py:build_agent_context` | Bounded chunks and concise summaries are documented in README. |
| Single-agent architecture | `run_question_agent` per question | Justification and Mermaid diagram in `docs/architecture.md`. |
| Evaluation harness | `backend/app/evaluation/` | `tests/test_evaluation_harness.py`; measured report in `docs/evaluation-results.md`. |
| Required metrics | `evaluation/metrics.py:aggregate_records` | Completion, tool correctness, trajectory length, taxonomy, and tokens are reported. |
| Token accounting | Provider usage extraction and `TokenUsage` | `tests/test_token_accounting.py` verifies 100+20 planner plus 200+30 grade = 350 total. |
| Failure injection | `evaluation/failure_injection.py:failing_retrieval` | Retrieval timeout is recorded as a contained soft failure. |
| Skill vs Agent | README Week 16 section | Deterministic capabilities versus decision controller is stated. |
| Tool boundary | `run_question_agent` and deterministic service persistence | README and architecture diagram document the boundary. |
| Existing Week 15 behavior | Grading service compatibility path and existing tests | Full pytest suite passes. |