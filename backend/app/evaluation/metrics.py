from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationRecord:
    case_id: str
    outcome: str
    completed_correctly: bool
    correct_tool_calls: int
    total_tool_calls: int
    steps: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    token_usage_available: bool
    failure_class: str | None = None
    failure_description: str | None = None
    trajectory: list[str] | None = None


def aggregate_records(records: list[EvaluationRecord]) -> dict:
    count = len(records)
    available = [record for record in records if record.token_usage_available]
    return {
        "case_count": count,
        "task_completion_rate": sum(record.completed_correctly for record in records) / count if count else 0.0,
        "correct_tool_calls": sum(record.correct_tool_calls for record in records),
        "total_tool_calls": sum(record.total_tool_calls for record in records),
        "tool_call_correctness": (
            sum(record.correct_tool_calls for record in records) / sum(record.total_tool_calls for record in records)
            if sum(record.total_tool_calls for record in records)
            else 1.0
        ),
        "average_trajectory_length": sum(record.steps for record in records) / count if count else 0.0,
        "token_usage_available_cases": len(available),
        "input_tokens": sum(record.input_tokens or 0 for record in available) if available else None,
        "output_tokens": sum(record.output_tokens or 0 for record in available) if available else None,
        "total_tokens": sum(record.total_tokens or 0 for record in available) if available else None,
        "hard_failures": sum(record.failure_class == "Hard failure" for record in records),
        "soft_failures": sum(record.failure_class == "Soft failure" for record in records),
        "cascading_soft_failures": sum(record.failure_class == "Cascading soft failure" for record in records),
    }

