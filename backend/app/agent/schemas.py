from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.schemas.grading import GradeEvaluation


class AgentAction(str, Enum):
    RETRIEVE_CONTEXT = "retrieve_context"
    GRADE_ANSWER = "grade_answer"
    VERIFY_GRADE = "verify_grade"
    REQUEST_CLARIFICATION = "request_clarification"
    FINISH = "finish"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    available: bool = False
    source: str | None = None

    def add(self, usage: dict | None) -> None:
        if not usage:
            return
        has_counts = any(
            key in usage
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
        )
        self.prompt_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        self.completion_tokens += int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        if not usage.get("total_tokens"):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.available = self.available or bool(has_counts and usage.get("available", True))
        self.source = self.source or usage.get("source")


class GradeVerification(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggested_next_step: str = "finish"


class AgentDecision(BaseModel):
    action: AgentAction
    query: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=8)
    reason: str = ""
    clarification_reason: str | None = None


class ToolCallRecord(BaseModel):
    iteration: int
    action: str
    arguments: dict = Field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    error: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class AgentState(BaseModel):
    question_number: str
    question_text: str
    max_marks: float
    model_answer: str
    mapped_student_answer: str
    flagged_ocr_text: str | None = None
    reference_set_id: str | None = None
    retrieved_chunks: list[dict] = Field(default_factory=list)
    current_grade: GradeEvaluation | None = None
    verification: GradeVerification | None = None
    previous_action_summaries: list[str] = Field(default_factory=list)
    current_iteration: int = 0
    status: str = "running"
    clarification_reason: str | None = None
    trajectory: list[ToolCallRecord] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
