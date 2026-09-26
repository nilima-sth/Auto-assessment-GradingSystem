from __future__ import annotations

from dataclasses import asdict

from backend.app.agent.schemas import AgentState
from backend.app.evaluation.metrics import EvaluationRecord


def build_agent_trace(*, case_id: str, prompt_version: str, state: AgentState) -> dict:
    """Serialize existing explicit agent telemetry without inferring hidden reasoning."""
    steps = [
        {
            "step": step_number,
            "iteration": record.iteration,
            "action": record.action,
            "arguments": record.arguments,
            "result_summary": record.result_summary,
            "success": record.success,
            "error": record.error,
            "token_usage": record.token_usage.model_dump(),
        }
        for step_number, record in enumerate(state.trajectory, start=1)
    ]
    termination_reason = "completed" if state.status == "completed" else state.clarification_reason
    return {
        "case_id": case_id,
        "prompt_version": prompt_version,
        "question_number": state.question_number,
        "status": state.status,
        "termination_reason": termination_reason,
        "total_iterations": state.current_iteration,
        "token_usage": state.token_usage.model_dump(),
        "retrieved_evidence_count": len(state.retrieved_chunks),
        "final_result": state.current_grade.model_dump() if state.current_grade else None,
        "steps": steps,
    }


def evaluation_record_payload(records: list[EvaluationRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def select_representative_traces(traces: list[dict], limit: int = 3) -> list[dict]:
    """Select distinct safe traces for an artifact without deriving new telemetry."""
    selected: list[dict] = []
    selected_case_ids: set[str] = set()
    criteria = (
        ("successful_terminal_state", lambda trace: trace["status"] == "completed"),
        (
            "recorded_tool_or_provider_error",
            lambda trace: any(not step["success"] for step in trace["steps"]),
        ),
        ("manual_review_terminal_state", lambda trace: trace["status"] == "manual_review"),
    )
    for reason, matches in criteria:
        trace = next(
            (item for item in traces if item["case_id"] not in selected_case_ids and matches(item)),
            None,
        )
        if trace is not None:
            selected.append({"selection_reason": reason, "trace": trace})
            selected_case_ids.add(trace["case_id"])
        if len(selected) >= limit:
            return selected
    for trace in traces:
        if trace["case_id"] not in selected_case_ids:
            selected.append({"selection_reason": "additional_distinct_case", "trace": trace})
            selected_case_ids.add(trace["case_id"])
        if len(selected) >= limit:
            break
    return selected
