from dataclasses import dataclass

import pytest

from backend.app.agent.loop import run_question_agent
from backend.app.agent.schemas import AgentDecision, AgentState, GradeVerification
from backend.app.rag.schemas import RagContext, RetrievedChunk
from backend.app.schemas.grading import GradeEvaluation


@dataclass
class Planner:
    decisions: list[AgentDecision]
    verifications: list[GradeVerification] | None = None

    def __post_init__(self):
        self.contexts = []
        self.verifications = self.verifications or [GradeVerification(passed=True)]

    def plan_agent_action(self, context):
        self.contexts.append(context)
        return self.decisions.pop(0), None

    def verify_grade(self, context):
        return self.verifications.pop(0), None


def _run(planner, retrieve_fn=None, grade_fn=None, max_iterations=5, **kwargs):
    return run_question_agent(
        question_number="Q1",
        question_text="Explain indexing.",
        max_marks=5,
        model_answer="Indexes speed lookup.",
        mapped_student_answer="Indexes speed lookup.",
        flagged_ocr_text=None,
        reference_set_id="refs" if retrieve_fn else None,
        planner=planner,
        grade_fn=grade_fn or (lambda *args, **kwargs: GradeEvaluation(
            question_number="Q1", awarded_marks=4, max_marks=5, feedback="Good"
        )),
        retrieve_fn=retrieve_fn,
        max_iterations=max_iterations,
        **kwargs,
    )


def test_agent_executes_multiple_iterations_and_finishes():
    planner = Planner([
        AgentDecision(action="grade_answer"),
        AgentDecision(action="verify_grade"),
        AgentDecision(action="finish"),
    ])
    state = _run(planner)
    assert state.status == "completed"
    assert [record.action for record in state.trajectory] == ["grade_answer", "verify_grade", "finish"]


def test_different_intermediate_results_create_different_trajectories():
    first = _run(Planner([AgentDecision(action="grade_answer"), AgentDecision(action="finish")]))
    second = _run(Planner([AgentDecision(action="request_clarification", reason="Unreadable")]))
    assert first.status == "completed"
    assert second.status == "manual_review"
    assert [item.action for item in first.trajectory] != [item.action for item in second.trajectory]


def test_retrieval_can_repeat_with_rewritten_query():
    queries = []

    def retrieve(**kwargs):
        queries.append(kwargs["query"])
        return RagContext(
            reference_set_id="refs",
            chunks=[RetrievedChunk(text="Evidence", source="ref.txt", chunk_index=len(queries), reference_set_id="refs")],
        )

    planner = Planner([
        AgentDecision(action="retrieve_context", query="first query"),
        AgentDecision(action="retrieve_context", query="rewritten query", top_k=2),
        AgentDecision(action="grade_answer"),
        AgentDecision(action="finish"),
    ])
    state = _run(planner, retrieve_fn=retrieve)
    assert state.status == "completed"
    assert queries == ["first query", "rewritten query"]


def test_agent_can_grade_and_finish_without_retrieval():
    state = _run(Planner([AgentDecision(action="grade_answer"), AgentDecision(action="finish")]))
    assert state.status == "completed"
    assert all(record.action != "retrieve_context" for record in state.trajectory)


def test_invalid_output_is_safe_manual_review():
    class InvalidPlanner:
        def plan_agent_action(self, context):
            return AgentDecision.model_validate({"action": "not_allowed"}), None

        def verify_grade(self, context):
            raise AssertionError("should not verify")

    state = _run(InvalidPlanner())
    assert state.status == "manual_review"
    assert state.current_grade is None
    assert state.trajectory[0].success is False


def test_max_iterations_does_not_return_confident_grade():
    state = _run(Planner([AgentDecision(action="grade_answer")]), max_iterations=1)
    assert state.status == "manual_review"
    assert "maximum" in state.clarification_reason.lower()
    assert state.current_grade is not None


def test_tool_exception_does_not_produce_confident_grade():
    def failing_retrieve(**kwargs):
        raise RuntimeError("vector store unavailable")

    state = _run(
        Planner([AgentDecision(action="retrieve_context")]),
        retrieve_fn=failing_retrieve,
    )
    assert state.status == "manual_review"
    assert state.current_grade is None
    assert state.trajectory[0].success is False
