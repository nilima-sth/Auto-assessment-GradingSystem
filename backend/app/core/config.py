from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    database_path: Path = Field(default=Path("../grades.db"), alias="DATABASE_PATH")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")
    poppler_path: str | None = Field(default=None, alias="POPPLER_PATH")
    trocr_model: str = Field(
        default="microsoft/trocr-base-handwritten",
        alias="TROCR_MODEL",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    answer_chunk_size: int = Field(default=220, alias="ANSWER_CHUNK_SIZE")
    answer_chunk_overlap: int = Field(default=80, alias="ANSWER_CHUNK_OVERLAP")
    similarity_threshold: float = Field(default=0.6, alias="SIMILARITY_THRESHOLD")
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, alias="LLM_TOP_P")
    llm_max_output_tokens: int | None = Field(default=None, alias="LLM_MAX_OUTPUT_TOKENS")
    llm_retry_attempts: int = Field(default=3, alias="LLM_RETRY_ATTEMPTS")
    llm_retry_base_delay: float = Field(default=0.5, alias="LLM_RETRY_BASE_DELAY")
    llm_tool_calling_enabled: bool = Field(default=True, alias="LLM_TOOL_CALLING_ENABLED")
    enable_vllm_fallback: bool = Field(default=True, alias="ENABLE_VLLM_FALLBACK")
    vllm_base_url: str = Field(default="http://localhost:8001/v1", alias="VLLM_BASE_URL")
    vllm_model: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", alias="VLLM_MODEL")
    vllm_timeout_seconds: float = Field(default=60.0, alias="VLLM_TIMEOUT_SECONDS")
    rate_limit_requests: int = Field(default=10, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    ocr_concurrency: int = Field(default=1, alias="OCR_CONCURRENCY")
    llm_concurrency: int = Field(default=3, alias="LLM_CONCURRENCY")
    chroma_path: Path = Field(default=Path("../data/chroma"), alias="CHROMA_PATH")
    rag_chunk_size: int = Field(default=700, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=100, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    agent_max_iterations: int = Field(default=5, alias="AGENT_MAX_ITERATIONS")

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def resolved_database_path(self) -> Path:
        if self.database_path.is_absolute():
            return self.database_path
        return (BACKEND_DIR / self.database_path).resolve()

    @property
    def resolved_chroma_path(self) -> Path:
        if self.chroma_path.is_absolute():
            return self.chroma_path
        return (BACKEND_DIR / self.chroma_path).resolve()

    @property
    def effective_poppler_path(self) -> str | None:
        return self.poppler_path or None

    @field_validator("poppler_path", "gemini_api_key", "llm_max_output_tokens", mode="before")
    @classmethod
    def empty_string_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
