from backend.app.core.config import get_settings
from backend.app.llm import vllm_provider


def test_vllm_provider_uses_openai_compatible_chat_endpoint(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setenv("VLLM_MODEL", "local-test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_TOP_P", "0.8")

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"question_number":"Q1","awarded_marks":4,'
                                '"max_marks":5,"feedback":"Good."}'
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(vllm_provider.httpx, "Client", FakeClient)

    result = vllm_provider.VLLMProvider().grade_answer("Q1", "Question?", "Reference.", "Student.", 5)

    assert captured["url"] == "http://vllm.test/v1/chat/completions"
    assert captured["json"]["model"] == "local-test-model"
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["top_p"] == 0.8
    assert result.awarded_marks == 4
    get_settings.cache_clear()
