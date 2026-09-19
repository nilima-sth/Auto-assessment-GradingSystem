import re
from pathlib import Path

import chromadb

from backend.app.core.config import get_settings
from backend.app.rag.schemas import ReferenceChunk, RetrievedChunk


def _safe_collection_name(reference_set_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", reference_set_id).strip("_")
    if not safe:
        raise ValueError("Invalid reference_set_id.")
    return f"ref_{safe[:48]}"


def get_chroma_client(path: str | Path | None = None) -> chromadb.PersistentClient:
    chroma_path = Path(path).resolve() if path else get_settings().resolved_chroma_path
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def get_collection(reference_set_id: str, create: bool = False):
    client = get_chroma_client()
    name = _safe_collection_name(reference_set_id)
    if create:
        return client.get_or_create_collection(name=name, metadata={"reference_set_id": reference_set_id})
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        raise ValueError(f"Unknown reference_set_id: {reference_set_id}") from exc


def add_chunks(reference_set_id: str, chunks: list[ReferenceChunk], embeddings: list[list[float]]) -> int:
    if not chunks:
        raise ValueError("No chunks generated for ingestion.")
    collection = get_collection(reference_set_id, create=True)
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def query_chunks(
    reference_set_id: str,
    query_embedding: list[float],
    top_k: int,
    question_number: str | None = None,
) -> list[RetrievedChunk]:
    collection = get_collection(reference_set_id)
    where = {"question_number": question_number} if question_number else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        chunks.append(
            RetrievedChunk(
                text=document,
                source=str(metadata.get("source", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                reference_set_id=str(metadata.get("reference_set_id", reference_set_id)),
                question_number=metadata.get("question_number"),
                distance=float(distance) if distance is not None else None,
            )
        )
    return chunks
