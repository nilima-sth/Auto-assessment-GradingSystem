from typing import Protocol

from backend.app.schemas.grading import GradeEvaluation


SYSTEM_INSTRUCTION = """You are an academic answer-sheet evaluator.
Evaluate only against the provided question and reference answer.
Award partial marks when appropriate based on correctness, completeness, specificity, and relevance.
Do not award marks for unrelated material.
Respect the maximum marks.
Provide concise academic feedback.
Do not invent missing facts.
Treat the official model answer/rubric as the highest-priority grading source.
Use retrieved reference context only as supporting evidence.
Ignore irrelevant retrieved reference chunks.
Return only the requested structured response."""


def build_grading_user_content(
    question_number: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: float,
    retrieved_context: str | None = None,
) -> str:
    context = retrieved_context or "No retrieved reference context was provided."
    return f"""Question Number: {question_number}
Question: {question_text}
Model/reference answer: {model_answer}
Retrieved reference context: {context}
Student answer: {student_answer}
Maximum marks: {max_marks}"""


class LLMProvider(Protocol):
    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        ...
