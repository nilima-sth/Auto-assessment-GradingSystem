from __future__ import annotations

import logging
import time
from typing import Callable

from backend.app.llm.base import LLMProvider
from backend.app.schemas.grading import GradeEvaluation

logger = logging.getLogger(__name__)


def is_retryable_llm_error(exc: Exception) -> bool:
    return not isinstance(exc, ValueError)


class RetryingLLMProvider:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        provider_name: str,
        attempts: int,
        base_delay: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.attempts = max(1, attempts)
        self.base_delay = max(0.0, base_delay)
        self.sleep = sleep

    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                evaluation = self.provider.grade_answer(
                    question_number,
                    question_text,
                    model_answer,
                    student_answer,
                    max_marks,
                    retrieved_context=retrieved_context,
                )
                logger.info("LLM provider used: %s", self.provider_name)
                return evaluation
            except Exception as exc:
                if not is_retryable_llm_error(exc):
                    raise
                last_error = exc
                if attempt == self.attempts:
                    break
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM provider %s failed on attempt %s/%s; retrying.",
                    self.provider_name,
                    attempt,
                    self.attempts,
                )
                if delay:
                    self.sleep(delay)
        assert last_error is not None
        raise RuntimeError(f"{self.provider_name} failed after {self.attempts} attempts.") from last_error


class FallbackLLMProvider:
    def __init__(self, primary: LLMProvider, fallback: LLMProvider, *, primary_name: str, fallback_name: str) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        try:
            return self.primary.grade_answer(
                question_number,
                question_text,
                model_answer,
                student_answer,
                max_marks,
                retrieved_context=retrieved_context,
            )
        except Exception as exc:
            if not is_retryable_llm_error(exc):
                raise
            logger.warning(
                "LLM provider %s failed; falling back to %s.",
                self.primary_name,
                self.fallback_name,
            )
            evaluation = self.fallback.grade_answer(
                question_number,
                question_text,
                model_answer,
                student_answer,
                max_marks,
                retrieved_context=retrieved_context,
            )
            logger.info("LLM provider used: %s", self.fallback_name)
            return evaluation
