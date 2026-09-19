from pydantic import BaseModel, Field, model_validator


class GradeEvaluation(BaseModel):
    question_number: str
    awarded_marks: float
    max_marks: float
    feedback: str

    @model_validator(mode="after")
    def clamp_awarded_marks(self) -> "GradeEvaluation":
        self.awarded_marks = max(0.0, min(float(self.awarded_marks), float(self.max_marks)))
        return self


class QuestionGrade(BaseModel):
    question_number: str
    awarded_marks: float
    max_marks: float
    feedback: str
    retrieved_chunks: list[dict] = Field(default_factory=list)


class StudentGradeResult(BaseModel):
    student: str
    questions: list[QuestionGrade]
    total_awarded: float
    total_possible: float
    flagged_text: str | None = None


class GradingResponse(BaseModel):
    students: list[StudentGradeResult] = Field(default_factory=list)
