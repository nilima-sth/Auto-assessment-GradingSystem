import pytest

from backend.app.llm.resilient_provider import FallbackLLMProvider, RetryingLLMProvider
from backend.app.schemas.grading import GradeEvaluation


class FlakyProvider:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def grade_answer(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary provider failure")
        return GradeEvaluation(question_number="Q1", awarded_marks=3, max_marks=5, feedback="OK")


def test_retrying_provider_retries_bounded_failures() -> None:
    provider = FlakyProvider(failures=2)
    retrying = RetryingLLMProvider(
        provider,
        provider_name="test",
        attempts=3,
        base_delay=0,
        sleep=lambda _delay: None,
    )

    result = retrying.grade_answer("Q1", "Q?", "A.", "S.", 5)

    assert provider.calls == 3
    assert result.awarded_marks == 3


def test_fallback_provider_uses_local_provider_after_primary_failure() -> None:
    primary = FlakyProvider(failures=10)
    fallback = FlakyProvider(failures=0)
    provider = FallbackLLMProvider(primary, fallback, primary_name="gemini", fallback_name="vllm")

    result = provider.grade_answer("Q1", "Q?", "A.", "S.", 5)

    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.feedback == "OK"


def test_retrying_provider_does_not_retry_value_error() -> None:
    class BadInputProvider:
        calls = 0

        def grade_answer(self, *args, **kwargs):
            self.calls += 1
            raise ValueError("bad input")

    provider = BadInputProvider()
    retrying = RetryingLLMProvider(
        provider,
        provider_name="test",
        attempts=3,
        base_delay=0,
        sleep=lambda _delay: None,
    )

    with pytest.raises(ValueError):
        retrying.grade_answer("Q1", "Q?", "A.", "S.", 5)
    assert provider.calls == 1
