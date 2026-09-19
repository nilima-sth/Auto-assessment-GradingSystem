from types import SimpleNamespace

from backend.app.core.config import get_settings
from backend.app.rag import ingestion


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_txt_ingestion_creates_chunks_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    monkeypatch.setattr(ingestion, "get_embedder", lambda: FakeEmbedder())

    exam = tmp_path / "exam.txt"
    exam.write_text("Q1: What is testing? (5)\nA1: Testing checks software behavior.", encoding="utf-8")
    reference = tmp_path / "notes.txt"
    reference.write_text("Testing includes unit tests, integration tests, and acceptance tests.", encoding="utf-8")

    result = ingestion.ingest_reference_set(exam, [reference], reference_set_id="unit-rag")

    assert result.reference_set_id == "unit-rag"
    assert result.documents_ingested == 2
    assert result.chunks_indexed >= 2

    collection = ingestion.add_chunks.__globals__["get_collection"]("unit-rag")
    stored = collection.get(include=["metadatas", "documents"])
    assert stored["documents"]
    assert any(metadata["reference_set_id"] == "unit-rag" for metadata in stored["metadatas"])
    assert any(metadata.get("question_number") == "Q1" for metadata in stored["metadatas"])

    get_settings.cache_clear()


def test_chunk_reference_text_uses_overlap(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "10")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "3")
    get_settings.cache_clear()

    chunks = ingestion.chunk_reference_text("abcdefghijklmnopqrstuvwxyz")

    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("hij")

    get_settings.cache_clear()
