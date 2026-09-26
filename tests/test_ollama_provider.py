from types import SimpleNamespace

from backend.app.core.config import get_settings
from backend.app.llm.ollama_provider import OllamaProvider


def test_ollama_provider_requests_json_without_terminal_word_wrapping(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=b'{"question_number":"Q1","awarded_marks":4,"max_marks":5,"feedback":"Good."}',
            stderr=b"",
        )

    monkeypatch.setattr("backend.app.llm.ollama_provider.subprocess.run", fake_run)
    result = OllamaProvider().grade_answer("Q1", "Question", "Reference", "Student", 5)

    assert result.awarded_marks == 4
    assert captured["command"] == [
        "ollama", "run", "test-model", "--format", "json", "--nowordwrap", "--hidethinking"
    ]
    get_settings.cache_clear()
