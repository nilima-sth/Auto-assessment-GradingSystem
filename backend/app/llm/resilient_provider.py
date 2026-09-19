from __future__ import annotations

import logging
import time
from typing import Callable

from backend.app.llm.base import LLMProvider
from backend.app.schemas.grading import GradeEvaluation

logger = logging.getLogger(__name__)


def _merge_usage(*usages: dict | None) -> dict | None:
    present = [usage for usage in usages if usage]
    if not present:
        return None
    return {
        "prompt_tokens": sum(int(item.get("prompt_tokens", item.get("input_tokens", 0)) or 0) for item in present),
        "completion_tokens": sum(int(item.get("completion_tokens", item.get("output_tokens", 0)) or 0) for item in present),
        "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in present),
        "source": "+".join(dict.fromkeys(str(item.get("source", "provider")) for item in present)),
    }


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
        self._last_usage: dict | None = None

    def consume_last_usage(self) -> dict | None:
        usage = self._last_usage
        self._last_usage = None
        return usage

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
        usages: list[dict] = []
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
                usage = getattr(self.provider, "consume_last_usage", lambda: None)()
                if usage:
                    usages.append(usage)
                self._last_usage = _merge_usage(*usages)
                logger.info("LLM provider used: %s", self.provider_name)
                return evaluation
            except Exception as exc:
                usage = getattr(self.provider, "consume_last_usage", lambda: None)()
                if usage:
                    usages.append(usage)
                self._last_usage = _merge_usage(*usages)
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

    def plan_agent_action(self, context: str):
        return self.provider.plan_agent_action(context)

    def verify_grade(self, context: str):
        return self.provider.verify_grade(context)


class FallbackLLMProvider:
    def __init__(self, primary: LLMProvider, fallback: LLMProvider, *, primary_name: str, fallback_name: str) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._last_usage: dict | None = None

    def consume_last_usage(self) -> dict | None:
        usage = self._last_usage
        self._last_usage = None
        return usage

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
            evaluation = self.primary.grade_answer(
                question_number,
                question_text,
                model_answer,
                student_answer,
                max_marks,
                retrieved_context=retrieved_context,
            )
            self._last_usage = getattr(self.primary, "consume_last_usage", lambda: None)()
            return evaluation
        except Exception as exc:
            if not is_retryable_llm_error(exc):
                raise
            primary_usage = getattr(self.primary, "consume_last_usage", lambda: None)()
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
            fallback_usage = getattr(self.fallback, "consume_last_usage", lambda: None)()
            self._last_usage = _merge_usage(primary_usage, fallback_usage)
            logger.info("LLM provider used: %s", self.fallback_name)
        return evaluation

    def plan_agent_action(self, context: str):
        try:
            return self.primary.plan_agent_action(context)
        except Exception:
            return self.fallback.plan_agent_action(context)

    def verify_grade(self, context: str):
        try:
            return self.primary.verify_grade(context)
        except Exception:
            return self.fallback.verify_grade(context)
