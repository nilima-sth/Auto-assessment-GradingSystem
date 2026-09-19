from types import SimpleNamespace

from backend.app.core.config import get_settings
from backend.app.llm import gemini_provider


def test_gemini_provider_validates_mocked_structured_response(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                parsed={
                    "question_number": "Q1",
                    "awarded_marks": 99,
                    "max_marks": 99,
                    "feedback": "Strong answer.",
                },
                text="",
            )

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(gemini_provider.genai, "Client", FakeClient)

    provider = gemini_provider.GeminiProvider()
    grade = provider.grade_answer("Q1", "Question?", "Reference.", "Student.", 5)

    assert grade.question_number == "Q1"
    assert grade.awarded_marks == 5
    assert grade.max_marks == 5
    assert grade.feedback == "Strong answer."
    assert captured["config"].response_mime_type == "application/json"

    get_settings.cache_clear()
