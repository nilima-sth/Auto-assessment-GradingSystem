from pathlib import Path
import asyncio

from fastapi import UploadFile

from backend.app.api import grading as grading_api
from backend.app.api import rag as rag_api
from backend.app.rag.schemas import IngestionResult


def test_rag_ingest_endpoint_uses_service(tmp_path, monkeypatch) -> None:
    exam_path = tmp_path / "exam.txt"
    exam_path.write_text("Q1: Test? (5)\nA1: Reference.", encoding="utf-8")
    reference_path = tmp_path / "notes.txt"
    reference_path.write_text("Extra notes.", encoding="utf-8")

    captured = {}

    def fake_ingest_reference_set(exam_path, reference_paths):
        captured["exam_exists"] = Path(exam_path).exists()
        captured["reference_count"] = len(reference_paths)
        return IngestionResult(reference_set_id="abc123", documents_ingested=2, chunks_indexed=3)

    monkeypatch.setattr(rag_api, "ingest_reference_set", fake_ingest_reference_set)

    with exam_path.open("rb") as exam_file, reference_path.open("rb") as reference_file:
        result = asyncio.run(
            rag_api.ingest(
                exam_file=UploadFile(filename="exam.txt", file=exam_file),
                reference_files=[UploadFile(filename="notes.txt", file=reference_file)],
            )
        )

    assert result.reference_set_id == "abc123"
    assert captured == {"exam_exists": True, "reference_count": 1}


def test_grading_endpoint_accepts_reference_set_id(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "grades.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    exam_path = tmp_path / "exam.txt"
    exam_path.write_text("Q1: Test? (5)\nA1: Reference.", encoding="utf-8")
    student_path = tmp_path / "student.pdf"
    student_path.write_bytes(b"%PDF-1.4\n")

    captured = {}

    async def fake_grade_batch_async(exam_path, student_paths, reference_set_id=None):
        captured["reference_set_id"] = reference_set_id
        return {
            "student.pdf": {
                "Q1": {
                    "awarded_marks": 4,
                    "max_marks": 5,
                    "feedback": "Good.",
                    "retrieved_chunks": [{"source": "notes.txt", "chunk_index": 0}],
                }
            }
        }

    monkeypatch.setattr(grading_api, "grade_batch_async", fake_grade_batch_async)

    with exam_path.open("rb") as exam_file, student_path.open("rb") as student_file:
        response = asyncio.run(
            grading_api.evaluate(
                exam_file=UploadFile(filename="exam.txt", file=exam_file),
                student_files=[UploadFile(filename="student.pdf", file=student_file)],
                reference_set_id="abc123",
            )
        )

    assert captured["reference_set_id"] == "abc123"
    assert response.students[0].questions[0].retrieved_chunks[0]["source"] == "notes.txt"
