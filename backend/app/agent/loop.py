from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from backend.app.agent.context import build_agent_context
from backend.app.agent.schemas import (
    AgentDecision,
    AgentState,
    GradeVerification,
    TokenUsage,
    ToolCallRecord,
)
from backend.app.rag.retrieval import retrieve_rag_context
from backend.app.schemas.grading import GradeEvaluation


class AgentPlanner(Protocol):
    def plan_agent_action(self, context: str) -> tuple[AgentDecision, dict | None]: ...

    def verify_grade(self, context: str) -> tuple[GradeVerification, dict | None]: ...


def _consume_provider_usage(planner: AgentPlanner) -> dict | None:
    consume = getattr(planner, "consume_last_usage", None)
    return consume() if consume else None


def _summary(action: str, result: object) -> str:
    if action == "retrieve_context":
        return f"Retrieved {len(getattr(result, 'chunks', []))} evidence chunks."
    if action == "grade_answer":
        return f"Graded {getattr(result, 'awarded_marks', '?')}/{getattr(result, 'max_marks', '?')}."
    if action == "verify_grade":
        return f"Verification passed={getattr(result, 'passed', False)}."
    return str(result)[:240]


def run_question_agent(
    *,
    question_number: str,
    question_text: str,
    max_marks: float,
    model_answer: str,
    mapped_student_answer: str,
    flagged_ocr_text: str | None,
    reference_set_id: str | None,
    planner: AgentPlanner,
    grade_fn: Callable[..., GradeEvaluation],
    retrieve_fn: Callable[..., object] = retrieve_rag_context,
    max_iterations: int = 5,
) -> AgentState:
    state = AgentState(
        question_number=question_number,
        question_text=question_text,
        max_marks=max_marks,
        model_answer=model_answer,
        mapped_student_answer=mapped_student_answer,
        flagged_ocr_text=flagged_ocr_text,
        reference_set_id=reference_set_id,
    )

    for iteration in range(1, max(1, max_iterations) + 1):
        state.current_iteration = iteration
        decision = None
        usage = None
        for planner_attempt in range(2):
            try:
                decision, usage = planner.plan_agent_action(build_agent_context(state))
                state.token_usage.add(usage)
                break
            except Exception as exc:
                state.trajectory.append(
                    ToolCallRecord(
                        iteration=iteration,
                        action="planner",
                        arguments={"attempt": planner_attempt + 1},
                        success=False,
                        error=str(exc),
                    )
                )
                if planner_attempt == 1:
                    state.status = "manual_review"
                    state.clarification_reason = f"Planner returned invalid or unusable output: {exc}"
        if decision is None:
            state.status = "manual_review"
            break
        record = ToolCallRecord(
            iteration=iteration,
            action=decision.action.value,
            arguments=decision.model_dump(exclude_none=True),
            token_usage=TokenUsage(),
        )
        record.token_usage.add(usage)

        try:
            action = decision.action.value
            if action == "retrieve_context":
                if not reference_set_id:
                    raise ValueError("No reference set is available for retrieval.")
                top_k = decision.top_k or 4
                rag_context = retrieve_fn(
                    reference_set_id=reference_set_id,
                    question_number=question_number,
                    question_text=question_text,
                    student_answer=mapped_student_answer,
                    query=decision.query,
                    top_k=top_k,
                )
                state.retrieved_chunks = [chunk.model_dump() for chunk in rag_context.chunks]
                record.result_summary = _summary(action, rag_context)
            elif action == "grade_answer":
                state.current_grade = grade_fn(
                    question_number=question_number,
                    question_text=question_text,
                    model_answer=model_answer,
                    student_answer=mapped_student_answer,
                    max_marks=max_marks,
                    retrieved_context=(
                        "\n\n".join(
                            f"[{item.get('source')}#{item.get('chunk_index')}] {item.get('text', '')}"
                            for item in state.retrieved_chunks
                        )
                        or None
                    ),
                )
                grading_usage = _consume_provider_usage(planner)
                state.token_usage.add(grading_usage)
                record.token_usage.add(grading_usage)
                record.result_summary = _summary(action, state.current_grade)
            elif action == "verify_grade":
                if state.current_grade is None:
                    raise ValueError("Cannot verify before a grade exists.")
                verification, usage = planner.verify_grade(build_agent_context(state))
                state.token_usage.add(usage)
                record.token_usage.add(usage)
                state.verification = verification
                record.result_summary = _summary(action, verification)
            elif action == "request_clarification":
                state.status = "manual_review"
                state.clarification_reason = decision.clarification_reason or decision.reason or "Agent requested manual review."
                record.result_summary = state.clarification_reason
            elif action == "finish":
                if state.current_grade is None:
                    raise ValueError("Agent attempted to finish without a grade.")
                state.status = "completed"
                record.result_summary = _summary(action, state.current_grade)
            else:
                raise ValueError(f"Unsupported agent action: {action}")
            state.previous_action_summaries.append(record.result_summary)
            state.trajectory.append(record)
        except Exception as exc:
            failed_usage = _consume_provider_usage(planner)
            state.token_usage.add(failed_usage)
            record.token_usage.add(failed_usage)
            record.success = False
            record.error = str(exc)
            state.trajectory.append(record)
            state.status = "manual_review"
            state.clarification_reason = f"Agent tool failed during {decision.action.value}: {exc}"

        if state.status in {"completed", "manual_review"}:
            break

    if state.status == "running":
        state.status = "manual_review"
        state.clarification_reason = (
            f"Agent reached the maximum of {max(1, max_iterations)} iterations without safely finishing."
        )
    return state
