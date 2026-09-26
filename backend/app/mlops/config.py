from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIRECTORY = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Versioned metadata for a reproducible experiment baseline."""

    version: str
    description: str
    system_prompt_id: str
    system_prompt_path: Path
    planner_prompt_id: str
    planner_prompt_path: Path
    evaluation_mode: str

    def prompt_text(self) -> str:
        return self.system_prompt_path.read_text(encoding="utf-8")

    def planner_prompt_text(self) -> str:
        return self.planner_prompt_path.read_text(encoding="utf-8")


def load_experiment_config(version: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Configuration version may contain only letters, numbers, hyphens, and underscores.")
    config_path = CONFIG_DIRECTORY / f"{version}.json"
    if not config_path.is_file():
        raise ValueError(f"Unknown experiment configuration: {version}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_path = PROJECT_ROOT / data["system_prompt_path"]
    planner_prompt_path = PROJECT_ROOT / data["planner_prompt_path"]
    if not prompt_path.is_file():
        raise ValueError(f"Configured prompt file does not exist: {prompt_path}")
    if not planner_prompt_path.is_file():
        raise ValueError(f"Configured planner prompt file does not exist: {planner_prompt_path}")
    return ExperimentConfig(
        version=str(data["version"]),
        description=str(data["description"]),
        system_prompt_id=str(data["system_prompt_id"]),
        system_prompt_path=prompt_path,
        planner_prompt_id=str(data["planner_prompt_id"]),
        planner_prompt_path=planner_prompt_path,
        evaluation_mode=str(data["evaluation_mode"]),
    )


def configured_model(settings: Settings) -> str:
    provider = settings.llm_provider.lower().strip()
    if provider == "gemini":
        return settings.gemini_model
    if provider == "vllm":
        return settings.vllm_model
    return settings.ollama_model


def resolved_parameters(config: ExperimentConfig, settings: Settings) -> dict[str, str | int | float | bool]:
    """Return only settings that are actually used by the current grading system."""
    return {
        "prompt_version": config.version,
        "system_prompt_id": config.system_prompt_id,
        "planner_prompt_id": config.planner_prompt_id,
        "evaluation_mode": config.evaluation_mode,
        "configured_provider": settings.llm_provider,
        "configured_model": configured_model(settings),
        "llm_temperature": settings.llm_temperature,
        "llm_top_p": settings.llm_top_p,
        "llm_max_output_tokens": settings.llm_max_output_tokens or "unset",
        "llm_tool_calling_enabled": settings.llm_tool_calling_enabled,
        "enable_vllm_fallback": settings.enable_vllm_fallback,
        "rag_top_k": settings.rag_top_k,
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
        "answer_chunk_size": settings.answer_chunk_size,
        "answer_chunk_overlap": settings.answer_chunk_overlap,
        "similarity_threshold": settings.similarity_threshold,
        "agent_max_iterations": settings.agent_max_iterations,
    }
