import json
import subprocess

from backend.app.core.config import get_settings
from backend.app.llm.base import SYSTEM_INSTRUCTION, build_grading_user_content
from backend.app.schemas.grading import GradeEvaluation


class OllamaProvider:
    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        prompt = f"""{SYSTEM_INSTRUCTION}

{build_grading_user_content(question_number, question_text, model_answer, student_answer, max_marks, retrieved_context)}

Respond as strict JSON with keys "question_number", "awarded_marks", "max_marks", and "feedback", nothing else.
JSON:"""

        settings = get_settings()
        proc = subprocess.run(
            ["ollama", "run", settings.ollama_model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            error = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(error or f"Ollama exited with code {proc.returncode}")

        raw = proc.stdout.decode("utf-8", errors="ignore").strip()
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            payload = raw[start : end + 1] if start != -1 and end != -1 else raw
            data = json.loads(payload)
            evaluation = GradeEvaluation.model_validate(data)
        except Exception:
            evaluation = GradeEvaluation(
                question_number=question_number,
                awarded_marks=0.0,
                max_marks=max_marks,
                feedback="Unclear - manual review required (invalid LLM JSON).",
            )

        return GradeEvaluation(
            question_number=question_number,
            awarded_marks=evaluation.awarded_marks,
            max_marks=max_marks,
            feedback=evaluation.feedback,
        )
