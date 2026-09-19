from backend.app.core.config import get_settings
from backend.app.rag.schemas import ReferenceChunk
from backend.app.rag.vector_store import add_chunks, query_chunks


def test_vector_store_persists_and_retrieves(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()

    chunks = [
        ReferenceChunk(
            id="set-a:doc:0",
            text="Operating systems manage hardware and services for programs.",
            metadata={"reference_set_id": "set-a", "source": "doc.txt", "chunk_index": 0},
        )
    ]
    add_chunks("set-a", chunks, [[1.0, 0.0, 0.0]])

    results = query_chunks("set-a", [1.0, 0.0, 0.0], top_k=1)

    assert results[0].reference_set_id == "set-a"
    assert "Operating systems" in results[0].text

    get_settings.cache_clear()


def test_reference_sets_are_isolated(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()

    add_chunks(
        "exam-a",
        [
            ReferenceChunk(
                id="exam-a:doc:0",
                text="Photosynthesis uses sunlight.",
                metadata={"reference_set_id": "exam-a", "source": "a.txt", "chunk_index": 0},
            )
        ],
        [[1.0, 0.0, 0.0]],
    )
    add_chunks(
        "exam-b",
        [
            ReferenceChunk(
                id="exam-b:doc:0",
                text="Operating systems manage hardware.",
                metadata={"reference_set_id": "exam-b", "source": "b.txt", "chunk_index": 0},
            )
        ],
        [[0.0, 1.0, 0.0]],
    )

    result = query_chunks("exam-a", [0.0, 1.0, 0.0], top_k=1)[0]

    assert result.reference_set_id == "exam-a"
    assert "Operating systems" not in result.text

    get_settings.cache_clear()
