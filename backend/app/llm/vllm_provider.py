import json

import httpx

from backend.app.core.config import get_settings
from backend.app.llm.base import SYSTEM_INSTRUCTION, build_grading_user_content
from backend.app.schemas.grading import GradeEvaluation


class VLLMProvider:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.vllm_base_url.rstrip("/")
        self.model = self.settings.vllm_model

    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        prompt = (
            build_grading_user_content(
                question_number,
                question_text,
                model_answer,
                student_answer,
                max_marks,
                retrieved_context,
            )
            + '\n\nRespond as strict JSON with keys "question_number", "awarded_marks", '
            '"max_marks", and "feedback", nothing else.'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "response_format": {"type": "json_object"},
        }
        if self.settings.llm_max_output_tokens:
            payload["max_tokens"] = self.settings.llm_max_output_tokens

        try:
            with httpx.Client(timeout=self.settings.vllm_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("vLLM OpenAI-compatible endpoint failed.") from exc

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        try:
            evaluation = GradeEvaluation.model_validate(json.loads(content))
        except Exception as exc:
            raise RuntimeError("vLLM returned invalid grading JSON.") from exc

        return GradeEvaluation(
            question_number=question_number,
            awarded_marks=evaluation.awarded_marks,
            max_marks=max_marks,
            feedback=evaluation.feedback,
        )
