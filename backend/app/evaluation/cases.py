from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.agent.schemas import AgentDecision, GradeVerification


@dataclass
class EvaluationCase:
    case_id: str
    description: str
    decisions: list[AgentDecision]
    expected_terminal_status: str
    required_actions: set[str] = field(default_factory=set)
    forbidden_actions: set[str] = field(default_factory=set)
    max_reasonable_steps: int = 5
    verification_results: list[GradeVerification] = field(default_factory=list)
    retrieval_failure: bool = False
    expected_failure_class: str | None = None


def build_deterministic_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="direct-grade",
            description="Sufficient answer completes without retrieval.",
            decisions=[AgentDecision(action="grade_answer"), AgentDecision(action="finish")],
            expected_terminal_status="completed",
            forbidden_actions={"retrieve_context", "request_clarification"},
        ),
        EvaluationCase(
            case_id="rag-needed",
            description="Agent retrieves evidence before grading.",
            decisions=[
                AgentDecision(action="retrieve_context", query="indexing reference"),
                AgentDecision(action="grade_answer"),
                AgentDecision(action="finish"),
            ],
            expected_terminal_status="completed",
            required_actions={"retrieve_context"},
        ),
        EvaluationCase(
            case_id="rewrite-retrieval",
            description="Initial evidence is inadequate and the planner rewrites the query.",
            decisions=[
                AgentDecision(action="retrieve_context", query="broad query"),
                AgentDecision(action="retrieve_context", query="precise rewritten query", top_k=2),
                AgentDecision(action="grade_answer"),
                AgentDecision(action="finish"),
            ],
            expected_terminal_status="completed",
            required_actions={"retrieve_context"},
        ),
        EvaluationCase(
            case_id="verification-replan",
            description="Verification fails and the planner chooses another action.",
            decisions=[
                AgentDecision(action="grade_answer"),
                AgentDecision(action="verify_grade"),
                AgentDecision(action="retrieve_context", query="supporting rubric"),
                AgentDecision(action="grade_answer"),
                AgentDecision(action="finish"),
            ],
            expected_terminal_status="completed",
            required_actions={"verify_grade", "retrieve_context"},
            verification_results=[GradeVerification(passed=False, issues=["Needs evidence"], suggested_next_step="retrieve_context")],
            max_reasonable_steps=5,
        ),
        EvaluationCase(
            case_id="ambiguous-review",
            description="Insufficient or ambiguous mapped answer requires manual review.",
            decisions=[AgentDecision(action="request_clarification", clarification_reason="OCR answer is ambiguous")],
            expected_terminal_status="manual_review",
            forbidden_actions={"finish"},
        ),
        EvaluationCase(
            case_id="retrieval-timeout",
            description="Injected retrieval timeout is contained as manual review.",
            decisions=[AgentDecision(action="retrieve_context", query="required evidence")],
            expected_terminal_status="manual_review",
            retrieval_failure=True,
            expected_failure_class="Soft failure",
        ),
    ]

