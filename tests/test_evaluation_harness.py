from backend.app.evaluation.harness import run_evaluation


def test_evaluation_harness_exercises_actual_agent_loop():
    records, aggregate = run_evaluation()

    assert len(records) == 6
    assert aggregate["task_completion_rate"] == 1.0
    assert aggregate["tool_call_correctness"] == 1.0
    timeout = next(record for record in records if record.case_id == "retrieval-timeout")
    assert timeout.outcome == "manual_review"
    assert timeout.failure_class == "Soft failure"

