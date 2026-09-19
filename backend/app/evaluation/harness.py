from __future__ import annotations

from dataclasses import dataclass

from backend.app.agent.loop import run_question_agent
from backend.app.agent.schemas import AgentDecision, GradeVerification
from backend.app.evaluation.cases import EvaluationCase, build_deterministic_cases
from backend.app.evaluation.failure_injection import failing_retrieval
from backend.app.evaluation.metrics import EvaluationRecord, aggregate_records
from backend.app.rag.schemas import RagContext, RetrievedChunk
from backend.app.schemas.grading import GradeEvaluation


@dataclass
class ScriptedPlanner:
    decisions: list[AgentDecision]
    verification_results: list[GradeVerification]

    def plan_agent_action(self, _context: str):
        return self.decisions.pop(0), None

    def verify_grade(self, _context: str):
        return self.verification_results.pop(0), None


def _retrieval(case_id: str, **kwargs) -> RagContext:
    query = kwargs.get("query") or "default query"
    return RagContext(
        reference_set_id="evaluation-refs",
        chunks=[
            RetrievedChunk(
                text=f"Evidence for {case_id}: {query}",
                source="evaluation-reference.txt",
                chunk_index=0,
                reference_set_id="evaluation-refs",
            )
        ],
    )


def _grade(*args, **kwargs) -> GradeEvaluation:
    return GradeEvaluation(
        question_number=kwargs["question_number"],
        awarded_marks=4,
        max_marks=kwargs["max_marks"],
        feedback="Deterministic evaluation grade.",
    )


def _failure_for(case: EvaluationCase, state) -> tuple[str | None, str | None]:
    failed_tool = next((record for record in state.trajectory if not record.success), None)
    if failed_tool is None and state.status == case.expected_terminal_status:
        return None, None
    if case.expected_failure_class:
        return case.expected_failure_class, failed_tool.error if failed_tool else state.clarification_reason
    if state.status != case.expected_terminal_status:
        return "Hard failure", f"Expected {case.expected_terminal_status}, got {state.status}."
    if failed_tool:
        return "Soft failure", failed_tool.error
    return None, None


def run_case(
    case: EvaluationCase,
    *,
    planner=None,
    grade_fn=None,
    retrieve_fn=None,
    reference_set_id: str = "evaluation-refs",
) -> EvaluationRecord:
    planner = planner or ScriptedPlanner(list(case.decisions), list(case.verification_results))
    retrieval = retrieve_fn or (
        failing_retrieval if case.retrieval_failure else lambda **kwargs: _retrieval(case.case_id, **kwargs)
    )
    state = run_question_agent(
        question_number="Q1",
        question_text="Explain indexing.",
        max_marks=5,
        model_answer="Indexes speed lookup.",
        mapped_student_answer="Indexes speed lookup." if case.case_id != "ambiguous-review" else "",
        flagged_ocr_text="Ambiguous OCR" if case.case_id == "ambiguous-review" else None,
        reference_set_id=reference_set_id,
        planner=planner,
        grade_fn=grade_fn or _grade,
        retrieve_fn=retrieval,
        max_iterations=case.max_reasonable_steps,
    )
    actions = [record.action for record in state.trajectory if record.action != "planner"]
    known_actions = {
        "retrieve_context", "grade_answer", "verify_grade", "request_clarification", "finish"
    }
    valid = all(action in known_actions for action in actions)
    behaviorally_allowed = not (set(actions) & case.forbidden_actions)
    required_present = case.required_actions.issubset(set(actions))
    within_limit = len(actions) <= case.max_reasonable_steps
    safe_expected_failure = bool(
        case.expected_failure_class
        and state.status == case.expected_terminal_status
        and state.current_grade is None
        and any(not record.success for record in state.trajectory)
    )
    correct = (
        state.status == case.expected_terminal_status
        and valid
        and behaviorally_allowed
        and required_present
        and within_limit
        and (all(record.success for record in state.trajectory if record.action != "planner") or safe_expected_failure)
    )
    correct_calls = sum(
        action in known_actions and action not in case.forbidden_actions
        for action in actions
    )
    usage = state.token_usage
    failure_class, failure_description = _failure_for(case, state)
    return EvaluationRecord(
        case_id=case.case_id,
        outcome=state.status,
        completed_correctly=correct,
        correct_tool_calls=correct_calls,
        total_tool_calls=len(actions),
        steps=len(actions),
        input_tokens=usage.prompt_tokens if usage.available else None,
        output_tokens=usage.completion_tokens if usage.available else None,
        total_tokens=usage.total_tokens if usage.available else None,
        token_usage_available=usage.available,
        failure_class=failure_class,
        failure_description=failure_description,
        trajectory=actions,
    )


def run_evaluation(
    cases: list[EvaluationCase] | None = None,
    *,
    planner_factory=None,
    grade_fn_factory=None,
    retrieve_fn=None,
    reference_set_id: str = "evaluation-refs",
) -> tuple[list[EvaluationRecord], dict]:
    records = [
        run_case(
            case,
            planner=planner_factory(case) if planner_factory else None,
            grade_fn=grade_fn_factory(case) if grade_fn_factory else None,
            retrieve_fn=retrieve_fn,
            reference_set_id=reference_set_id,
        )
        for case in (cases or build_deterministic_cases())
    ]
    return records, aggregate_records(records)


if __name__ == "__main__":
    records, aggregate = run_evaluation()
    for record in records:
        print(record)
    print(aggregate)
