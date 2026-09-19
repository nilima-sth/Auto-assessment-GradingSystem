from backend.app.core.config import get_settings
from backend.app.database.sqlite import init_db
from backend.app.llm.base import build_grading_user_content
from backend.app.rag.schemas import RagContext, RetrievedChunk
from backend.app.schemas.grading import GradeEvaluation
from backend.app.services import grading_service


def test_retrieved_context_enters_provider_prompt() -> None:
    prompt = build_grading_user_content(
        question_number="Q1",
        question_text="What is an operating system?",
        model_answer="It manages resources.",
        student_answer="It manages hardware.",
        max_marks=5,
        retrieved_context="[notes.txt#0] OS also manages software resources.",
    )

    assert "Retrieved reference context: [notes.txt#0] OS also manages software resources." in prompt


def test_grade_batch_passes_retrieved_context_to_provider(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "grades.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    get_settings.cache_clear()
    init_db()

    exam = tmp_path / "exam.txt"
    exam.write_text("Q1: What is an OS? (5)\nA1: It manages hardware and software.", encoding="utf-8")
    pdf = tmp_path / "student.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(grading_service, "ocr_pdf_to_text", lambda _: "An OS manages hardware.")
    monkeypatch.setattr(grading_service, "map_answers_to_questions", lambda questions, raw, **_: {"Q1": raw})
    monkeypatch.setattr(
        grading_service,
        "retrieve_rag_context",
        lambda **_: RagContext(
            reference_set_id="rag-1",
            chunks=[
                RetrievedChunk(
                    text="Operating systems also manage software resources.",
                    source="notes.txt",
                    chunk_index=0,
                    reference_set_id="rag-1",
                )
            ],
        ),
    )

    captured = {}

    class FakeProvider:
        def grade_answer(self, **kwargs):
            captured.update(kwargs)
            return GradeEvaluation(
                question_number=kwargs["question_number"],
                awarded_marks=4,
                max_marks=kwargs["max_marks"],
                feedback="Good.",
            )

    monkeypatch.setattr(grading_service, "get_llm_provider", lambda: FakeProvider())

    grading_service.grade_batch(str(exam), [str(pdf)], reference_set_id="rag-1")

    assert "Operating systems also manage software resources." in captured["retrieved_context"]

    get_settings.cache_clear()
