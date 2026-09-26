from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow

from backend.app.core.config import get_settings
from backend.app.evaluation.harness import run_evaluation_with_states
from backend.app.llm.base import SYSTEM_INSTRUCTION
from backend.app.llm.provider_factory import get_llm_provider
from backend.app.mlops.config import ExperimentConfig, load_experiment_config, resolved_parameters
from backend.app.mlops.traces import (
    build_agent_trace,
    evaluation_record_payload,
    select_representative_traces,
)
from backend.app.mlops.tracking import DEFAULT_EXPERIMENT_NAME, configure_local_tracking, numeric_metrics


@dataclass(frozen=True)
class ExperimentRunResult:
    run_id: str
    experiment_id: str
    tracking_uri: str
    summary: dict


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_v1_matches_current_prompt(config: ExperimentConfig) -> None:
    """Prevent the baseline artifact from silently drifting from Week 16 behavior."""
    if config.version == "v1" and config.prompt_text().strip() != SYSTEM_INSTRUCTION.strip():
        raise RuntimeError("Prompt v1 no longer matches backend.app.llm.base.SYSTEM_INSTRUCTION.")


def run_experiment(
    config_version: str = "v1",
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    execution_mode: str = "deterministic",
) -> ExperimentRunResult:
    """Track the Week 16 harness deterministically or through its configured Ollama provider."""
    if execution_mode not in {"deterministic", "ollama-live"}:
        raise ValueError("execution_mode must be 'deterministic' or 'ollama-live'.")
    config = load_experiment_config(config_version)
    _assert_v1_matches_current_prompt(config)
    settings = get_settings()
    if execution_mode == "ollama-live" and settings.llm_provider.lower().strip() != "ollama":
        raise ValueError("ollama-live execution requires LLM_PROVIDER=ollama.")
    parameters = resolved_parameters(config, settings)
    parameters["execution_mode"] = execution_mode
    effective_tracking_uri, experiment_id = configure_local_tracking(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    started = time.perf_counter()
    if execution_mode == "deterministic":
        records, aggregate, states = run_evaluation_with_states()
    else:
        case_providers = {}

        def planner_factory(case):
            provider = get_llm_provider(
                ollama_planner_prompt_template=config.planner_prompt_text(),
                ollama_system_instruction=config.prompt_text(),
            )
            case_providers[case.case_id] = provider
            return provider

        def grade_fn_factory(case):
            return case_providers[case.case_id].grade_answer

        records, aggregate, states = run_evaluation_with_states(
            planner_factory=planner_factory,
            grade_fn_factory=grade_fn_factory,
        )
    elapsed_seconds = time.perf_counter() - started
    traces = [
        build_agent_trace(case_id=record.case_id, prompt_version=config.version, state=state)
        for record, state in zip(records, states, strict=True)
    ]
    error_count = sum(
        any(not step.success for step in state.trajectory)
        for state in states
    )
    measured_metrics = {
        **aggregate,
        "average_agent_iterations": (
            sum(state.current_iteration for state in states) / len(states) if states else 0.0
        ),
        "error_count": error_count,
        "error_rate": error_count / len(states) if states else 0.0,
        "run_latency_seconds": elapsed_seconds,
        "trace_count": len(traces),
    }
    summary = {
        "config_version": config.version,
        "evaluation_mode": config.evaluation_mode,
        "execution_mode": execution_mode,
        "live_provider_execution": execution_mode == "ollama-live",
        "live_provider_status": (
            "executed: Ollama selected agent actions and grades; harness retrieval remains deterministic"
            if execution_mode == "ollama-live"
            else "not invoked: deterministic scripted harness only"
        ),
        "parameters": parameters,
        "evaluation_summary": aggregate,
        "measured_metrics": measured_metrics,
    }

    with mlflow.start_run(run_name=f"{config.version}-{execution_mode}") as active_run:
        mlflow.set_tags(
            {
                "prompt_version": config.version,
                "evaluation_mode": config.evaluation_mode,
                "execution_mode": execution_mode,
                "live_provider_execution": str(execution_mode == "ollama-live").lower(),
            }
        )
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_metrics(numeric_metrics(measured_metrics))
        with tempfile.TemporaryDirectory(prefix="week17_mlflow_") as temp_directory:
            artifact_directory = Path(temp_directory)
            _write_json(artifact_directory / "config_resolved.json", summary["parameters"])
            (artifact_directory / "prompt.txt").write_text(config.prompt_text(), encoding="utf-8")
            _write_json(artifact_directory / "evaluation_summary.json", summary)
            _write_json(artifact_directory / "evaluation_records.json", evaluation_record_payload(records))
            _write_json(artifact_directory / "agent_traces.json", traces)
            _write_json(
                artifact_directory / "representative_traces.json",
                select_representative_traces(traces),
            )
            mlflow.log_artifacts(str(artifact_directory))
        run_id = active_run.info.run_id

    return ExperimentRunResult(
        run_id=run_id,
        experiment_id=experiment_id,
        tracking_uri=effective_tracking_uri,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a versioned Week 16 evaluation experiment in MLflow.")
    parser.add_argument("--config", default="v1", help="Versioned experiment configuration to run (default: v1).")
    parser.add_argument("--tracking-uri", help="Optional MLflow tracking URI; defaults to local ./mlruns.")
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME}).",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("deterministic", "ollama-live"),
        default="deterministic",
        help="Run scripted evaluation only, or route planner/grader calls through configured Ollama.",
    )
    args = parser.parse_args()
    result = run_experiment(
        args.config,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        execution_mode=args.execution_mode,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "experiment_id": result.experiment_id,
                "tracking_uri": result.tracking_uri,
                "summary": result.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
