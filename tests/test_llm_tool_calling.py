from types import SimpleNamespace

from backend.app.core.config import get_settings
from backend.app.llm import gemini_provider


def test_gemini_model_selected_tool_call_is_executed(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_TOOL_CALLING_ENABLED", "true")

    captured_contents = []

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            captured_contents.append(kwargs["contents"])
            if self.calls == 1:
                function_call = SimpleNamespace(
                    name="get_question_context",
                    args={"question_number": "Q1"},
                )
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
                        )
                    ]
                )
            return SimpleNamespace(
                parsed={
                    "question_number": "Q1",
                    "awarded_marks": 4,
                    "max_marks": 5,
                    "feedback": "Good.",
                },
                text="",
            )

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(gemini_provider.genai, "Client", FakeClient)

    provider = gemini_provider.GeminiProvider()
    result = provider.grade_answer("Q1", "What is OS?", "Reference answer.", "Student answer.", 5)

    assert result.awarded_marks == 4
    assert "Gemini tool call get_question_context returned" in captured_contents[-1]
    get_settings.cache_clear()
