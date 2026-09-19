from backend.app.core.config import get_settings
from backend.app.rag.schemas import RagContext, RetrievedChunk
from backend.app.rag.vector_store import query_chunks
from backend.app.services.embedding_service import get_embedder


def retrieve_rag_context(
    reference_set_id: str,
    question_number: str,
    question_text: str,
    student_answer: str,
    top_k: int | None = None,
) -> RagContext:
    settings = get_settings()
    limit = top_k or settings.rag_top_k
    query = f"{question_text}\n{student_answer}".strip()
    query_embedding = get_embedder().encode([query])[0].tolist()

    chunks: list[RetrievedChunk] = []
    try:
        chunks.extend(query_chunks(reference_set_id, query_embedding, limit, question_number=question_number))
    except ValueError:
        raise
    except Exception:
        chunks = []

    if len(chunks) < limit:
        fallback_chunks = query_chunks(reference_set_id, query_embedding, limit)
        seen = {(chunk.source, chunk.chunk_index) for chunk in chunks}
        for chunk in fallback_chunks:
            key = (chunk.source, chunk.chunk_index)
            if key not in seen:
                chunks.append(chunk)
                seen.add(key)
            if len(chunks) >= limit:
                break

    return RagContext(reference_set_id=reference_set_id, chunks=chunks[:limit])
