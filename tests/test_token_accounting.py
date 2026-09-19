from backend.app.agent.loop import run_question_agent
from backend.app.agent.schemas import AgentDecision, GradeVerification
from backend.app.schemas.grading import GradeEvaluation


class UsagePlanner:
    def __init__(self):
        self.decisions = [AgentDecision(action="grade_answer"), AgentDecision(action="finish")]
        self.pending_grade_usage = None
        self.planner_calls = 0

    def plan_agent_action(self, _context):
        self.planner_calls += 1
        usage = (
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "source": "test"}
            if self.planner_calls == 1
            else None
        )
        return self.decisions.pop(0), usage

    def verify_grade(self, _context):
        return GradeVerification(passed=True), {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "source": "test"}

    def consume_last_usage(self):
        if self.pending_grade_usage is None:
            self.pending_grade_usage = {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230, "source": "test"}
            return self.pending_grade_usage
        return None


def test_question_usage_aggregates_planner_grade_and_verification_calls():
    state = run_question_agent(
        question_number="Q1",
        question_text="Question",
        max_marks=5,
        model_answer="Reference",
        mapped_student_answer="Answer",
        flagged_ocr_text=None,
        reference_set_id=None,
        planner=UsagePlanner(),
        grade_fn=lambda **kwargs: GradeEvaluation(
            question_number=kwargs["question_number"], max_marks=kwargs["max_marks"], awarded_marks=4, feedback="OK"
        ),
    )

    assert state.status == "completed"
    assert state.token_usage.available is True
    assert state.token_usage.prompt_tokens == 300
    assert state.token_usage.completion_tokens == 50
    assert state.token_usage.total_tokens == 350
