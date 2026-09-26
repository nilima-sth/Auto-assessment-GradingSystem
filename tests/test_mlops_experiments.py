import json

import mlflow
import pytest

from backend.app.evaluation.harness import run_evaluation_with_states
from backend.app.llm.base import SYSTEM_INSTRUCTION
from backend.app.mlops.config import load_experiment_config
from backend.app.mlops.experiments import run_experiment
from backend.app.mlops.traces import build_agent_trace, select_representative_traces


def test_v1_prompt_artifact_matches_existing_week16_prompt() -> None:
    config = load_experiment_config("v1")
    assert config.prompt_text().strip() == SYSTEM_INSTRUCTION.strip()
    assert "Choose one action" in config.planner_prompt_text()


def test_versioned_v2_uses_a_different_planner_prompt_only() -> None:
    v1 = load_experiment_config("v1")
    v2 = load_experiment_config("v2")

    assert v2.prompt_text() == v1.prompt_text()
    assert v2.planner_prompt_text() != v1.planner_prompt_text()


def test_versioned_v3_changes_grading_calibration_but_keeps_v2_planner() -> None:
    v2 = load_experiment_config("v2")
    v3 = load_experiment_config("v3")

    assert v3.planner_prompt_text() == v2.planner_prompt_text()
    assert v3.prompt_text() != v2.prompt_text()
    assert "award the full maximum marks" in v3.prompt_text()


def test_harness_exposes_existing_agent_states_for_trace_artifacts() -> None:
    records, aggregate, states = run_evaluation_with_states()

    assert len(records) == len(states) == aggregate["case_count"]
    trace = build_agent_trace(case_id=records[0].case_id, prompt_version="v1", state=states[0])
    assert trace["steps"][0]["action"] == "grade_answer"
    assert trace["final_result"]["awarded_marks"] == 4
    assert "reasoning" not in trace


def test_representative_trace_selection_uses_existing_trace_fields() -> None:
    records, _aggregate, states = run_evaluation_with_states()
    traces = [
        build_agent_trace(case_id=record.case_id, prompt_version="v1", state=state)
        for record, state in zip(records, states, strict=True)
    ]

    selected = select_representative_traces(traces)
    assert 1 <= len(selected) <= 3
    assert all("selection_reason" in item and "trace" in item for item in selected)
    assert len({item["trace"]["case_id"] for item in selected}) == len(selected)


def test_baseline_experiment_logs_metrics_and_artifacts(tmp_path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = run_experiment("v1", tracking_uri=tracking_uri, experiment_name="week17-test")

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result.run_id)
    assert run.data.params["prompt_version"] == "v1"
    assert run.data.params["evaluation_mode"] == "deterministic_scripted_harness"
    assert run.data.metrics["task_completion_rate"] == 1.0
    artifacts = {item.path for item in client.list_artifacts(result.run_id)}
    assert {
        "agent_traces.json",
        "config_resolved.json",
        "evaluation_records.json",
        "evaluation_summary.json",
        "prompt.txt",
        "representative_traces.json",
    } <= artifacts

    local_trace = client.download_artifacts(result.run_id, "agent_traces.json", str(tmp_path / "download"))
    traces = json.loads(open(local_trace, encoding="utf-8").read())
    assert len(traces) == 6
    assert any(trace["status"] == "manual_review" for trace in traces)


def test_live_experiment_requires_ollama_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ValueError, match="LLM_PROVIDER=ollama"):
        run_experiment("v1", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", execution_mode="ollama-live")
    get_settings.cache_clear()
