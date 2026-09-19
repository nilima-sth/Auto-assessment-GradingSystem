from pydantic import BaseModel, Field


class ReferenceChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, str | int]


class IngestionResult(BaseModel):
    reference_set_id: str
    documents_ingested: int
    chunks_indexed: int


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    reference_set_id: str
    question_number: str | None = None
    distance: float | None = None


class RagContext(BaseModel):
    reference_set_id: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"[{chunk.source}#{chunk.chunk_index}] {chunk.text}" for chunk in self.chunks
        )
