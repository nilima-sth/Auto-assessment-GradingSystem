from backend.app.core.config import get_settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.gemini_provider import GeminiProvider
from backend.app.llm.ollama_provider import OllamaProvider
from backend.app.llm.resilient_provider import FallbackLLMProvider, RetryingLLMProvider
from backend.app.llm.vllm_provider import VLLMProvider


def _with_retry(provider: LLMProvider, provider_name: str) -> LLMProvider:
    settings = get_settings()
    return RetryingLLMProvider(
        provider,
        provider_name=provider_name,
        attempts=settings.llm_retry_attempts,
        base_delay=settings.llm_retry_base_delay,
    )


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()
    if provider == "gemini":
        primary = _with_retry(GeminiProvider(), "gemini")
        if settings.enable_vllm_fallback:
            fallback = _with_retry(VLLMProvider(), "vllm")
            return FallbackLLMProvider(primary, fallback, primary_name="gemini", fallback_name="vllm")
        return primary
    if provider == "vllm":
        return _with_retry(VLLMProvider(), "vllm")
    if provider == "ollama":
        return _with_retry(OllamaProvider(), "ollama")
    raise ValueError("Unsupported LLM_PROVIDER. Use 'gemini', 'vllm', or 'ollama'.")
