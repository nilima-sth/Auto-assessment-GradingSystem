import json

import mlflow
import pytest

from backend.app.agent.schemas import AgentState, ToolCallRecord
from backend.app.schemas.grading import GradeEvaluation
from backend.app.mlops import regression


def test_golden_dataset_loads_and_has_fixed_valid_cases() -> None:
    cases = regression.load_golden_cases()

    assert 5 <= len(cases) <= 10
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.expected_status == "manual_review" for case in cases)
    assert any("retrieve_context" in case.required_actions for case in cases)


def test_golden_dataset_rejects_invalid_completed_case(tmp_path) -> None:
    invalid_case = {
        "case_id": "bad", "source": "test", "question_number": "Q1", "question_text": "q",
        "model_answer": "a", "student_answer": "a", "max_marks": 5, "expected_status": "completed",
        "required_actions": ["grade_answer"], "reference_response": "r", "retrieval_evidence": "e",
    }
    fixture = {"cases": [{**invalid_case, "case_id": f"bad-{index}"} for index in range(5)]}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_awarded_marks"):
        regression.load_golden_cases(path)


def test_project_sanity_explains_score_and_workflow_mismatches() -> None:
    case = regression.load_golden_cases()[0]
    state = AgentState(
        question_number=case.question_number, question_text=case.question_text, max_marks=case.max_marks,
        model_answer=case.model_answer, mapped_student_answer=case.student_answer, status="completed",
        current_grade=GradeEvaluation(question_number="Q1", awarded_marks=3, max_marks=5, feedback="partial"),
        trajectory=[ToolCallRecord(iteration=1, action="grade_answer")],
    )

    passed, explanation = regression._project_sanity(case, state)

    assert not passed
    assert "differ" in explanation


def test_regression_requires_ollama_before_invoking_agent(monkeypatch) -> None:
    class NonOllamaSettings:
        llm_provider = "gemini"

    monkeypatch.setattr(regression, "get_settings", lambda: NonOllamaSettings())
    with pytest.raises(ValueError, match="LLM_PROVIDER=ollama"):
        regression.run_regression("v1")


def test_regression_logs_fixed_input_and_judge_results(monkeypatch, tmp_path) -> None:
    class OllamaSettings:
        llm_provider = "ollama"
        ollama_model = "test-local-model"
        agent_max_iterations = 5
        llm_temperature = 0.2
        llm_top_p = 0.9
        llm_max_output_tokens = None
        llm_tool_calling_enabled = True
        enable_vllm_fallback = False
        rag_top_k = 4
        rag_chunk_size = 700
        rag_chunk_overlap = 100
        answer_chunk_size = 220
        answer_chunk_overlap = 80
        similarity_threshold = 0.6
        vllm_model = "unused"
        gemini_model = "unused"

    class FakeProvider:
        def grade_answer(self, **kwargs):
            return GradeEvaluation(question_number=kwargs["question_number"], awarded_marks=5, max_marks=5, feedback="ok")

        def plan_agent_action(self, _context):
            raise AssertionError("run_question_agent is mocked in this test")

    def fake_agent(**kwargs):
        return AgentState(
            question_number=kwargs["question_number"], question_text=kwargs["question_text"], max_marks=kwargs["max_marks"],
            model_answer=kwargs["model_answer"], mapped_student_answer=kwargs["mapped_student_answer"], status="completed",
            current_grade=GradeEvaluation(question_number=kwargs["question_number"], awarded_marks=5, max_marks=5, feedback="ok"),
            trajectory=[ToolCallRecord(iteration=1, action="grade_answer"), ToolCallRecord(iteration=2, action="finish")],
        )

    def fake_judges(rows, _model, report_path):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html>test report</html>", encoding="utf-8")
        return [{**row, "reference_correctness": "PASS", "workflow_adherence": "PASS",
                 "reference_correctness reasoning": "matches", "workflow_adherence reasoning": "matches"} for row in rows]

    monkeypatch.setattr(regression, "get_settings", lambda: OllamaSettings())
    monkeypatch.setattr(regression, "get_llm_provider", lambda **_kwargs: FakeProvider())
    monkeypatch.setattr(regression, "run_question_agent", fake_agent)
    monkeypatch.setattr(regression, "_run_evidently_judges", fake_judges)
    monkeypatch.setattr(regression, "REPORT_DIRECTORY", tmp_path / "reports")

    result = regression.run_regression("v1", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    assert result.case_count == 6
    assert result.report_path.endswith("v1_regression.html")
    assert result.run_id
    run = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}").get_run(result.run_id)
    assert run.data.metrics["pct_tests_passed"] == run.data.metrics["regression_pass_percentage"]
