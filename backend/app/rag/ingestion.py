import os
import re
import uuid
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rag.schemas import IngestionResult, ReferenceChunk
from backend.app.rag.vector_store import add_chunks
from backend.app.services.document_service import load_exam, read_pdf_text, read_text_file
from backend.app.services.embedding_service import get_embedder


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_reference_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    settings = get_settings()
    size = chunk_size or settings.rag_chunk_size
    step_back = overlap if overlap is not None else settings.rag_chunk_overlap
    normalized = normalize_text(text)
    if not normalized:
        return []
    chunks = []
    index = 0
    while index < len(normalized):
        chunk = normalized[index : index + size].strip()
        if chunk:
            chunks.append(chunk)
        index += max(size - step_back, 1)
    return chunks


def _read_reference_file(path: str | os.PathLike[str]) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return read_text_file(file_path)
    if suffix == ".pdf":
        return read_pdf_text(file_path)
    raise ValueError(f"Unsupported reference document type: {suffix or file_path.name}")


def _chunk_id(reference_set_id: str, source: str, chunk_index: int) -> str:
    return f"{reference_set_id}:{source}:{chunk_index}"


def _build_exam_chunks(reference_set_id: str, exam_path: str | os.PathLike[str]) -> list[ReferenceChunk]:
    questions, model_answers = load_exam(exam_path)
    source = Path(exam_path).name
    chunks = []
    for question_number, question in questions.items():
        text = (
            f"Question {question_number}: {question['text']}\n"
            f"Official model answer: {model_answers[question_number]}\n"
            f"Maximum marks: {question['marks']}"
        )
        for chunk_index, chunk_text in enumerate(chunk_reference_text(text)):
            chunks.append(
                ReferenceChunk(
                    id=_chunk_id(reference_set_id, f"exam-{question_number}", chunk_index),
                    text=chunk_text,
                    metadata={
                        "reference_set_id": reference_set_id,
                        "source": source,
                        "chunk_index": chunk_index,
                        "question_number": question_number,
                    },
                )
            )
    return chunks


def _build_reference_file_chunks(
    reference_set_id: str,
    file_path: str | os.PathLike[str],
    starting_index: int,
) -> list[ReferenceChunk]:
    source = Path(file_path).name
    text = _read_reference_file(file_path)
    chunks = []
    for offset, chunk_text in enumerate(chunk_reference_text(text)):
        chunk_index = starting_index + offset
        chunks.append(
            ReferenceChunk(
                id=_chunk_id(reference_set_id, source, chunk_index),
                text=chunk_text,
                metadata={
                    "reference_set_id": reference_set_id,
                    "source": source,
                    "chunk_index": chunk_index,
                },
            )
        )
    return chunks


def ingest_reference_set(
    exam_path: str | os.PathLike[str],
    reference_paths: list[str | os.PathLike[str]] | None = None,
    reference_set_id: str | None = None,
) -> IngestionResult:
    reference_id = reference_set_id or uuid.uuid4().hex
    chunks = _build_exam_chunks(reference_id, exam_path)
    documents_ingested = 1

    for reference_path in reference_paths or []:
        file_chunks = _build_reference_file_chunks(reference_id, reference_path, len(chunks))
        if not file_chunks:
            raise ValueError(f"Reference document is empty or produced no chunks: {Path(reference_path).name}")
        chunks.extend(file_chunks)
        documents_ingested += 1

    if not chunks:
        raise ValueError("No chunks generated for ingestion.")

    encoded = get_embedder().encode([chunk.text for chunk in chunks])
    embeddings = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    indexed = add_chunks(reference_id, chunks, embeddings)
    return IngestionResult(
        reference_set_id=reference_id,
        documents_ingested=documents_ingested,
        chunks_indexed=indexed,
    )
