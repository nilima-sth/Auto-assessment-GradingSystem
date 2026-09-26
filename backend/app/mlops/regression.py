"""Fixed golden regression evaluation for Week 17 Track B.

This is intentionally separate from the Week 16 harness metrics: it executes
versioned live agent configurations on fixed inputs, then evaluates their
candidate records against hand-reviewed reference decisions.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from evidently import Dataset, Report
from evidently.descriptors import LLMEval
from evidently.llm.models import LLMMessage
from evidently.llm.templates import BinaryClassificationPromptTemplate, Uncertainty
from evidently.llm.utils.wrapper import OllamaOptions
from evidently.metrics import CategoryCount, MeanValue, RowCount

from backend.app.agent.loop import run_question_agent
from backend.app.core.config import get_settings
from backend.app.llm.provider_factory import get_llm_provider
from backend.app.mlops.config import PROJECT_ROOT, load_experiment_config, resolved_parameters
from backend.app.mlops.tracking import configure_local_tracking, numeric_metrics
from backend.app.rag.schemas import RagContext, RetrievedChunk


GOLDEN_DATASET_PATH = Path(__file__).parent / "golden" / "regression_v1.json"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "evidently"
REGRESSION_EXPERIMENT_NAME = "week17-regression"


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    source: str
    question_number: str
    question_text: str
    model_answer: str
    student_answer: str
    max_marks: float
    expected_status: str
    expected_awarded_marks: float | None
    required_actions: tuple[str, ...]
    reference_response: str
    retrieval_evidence: str
    flagged_ocr_text: str | None = None
    max_tool_calls: int = 5


@dataclass(frozen=True)
class RegressionRunResult:
    run_id: str
    config_version: str
    case_count: int
    passed_cases: int
    pass_percentage: float
    report_path: str
    failures: list[str] = field(default_factory=list)


def load_golden_cases(path: Path = GOLDEN_DATASET_PATH) -> list[GoldenCase]:
    """Load and validate the project-owned, fixed regression dataset."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not 5 <= len(cases) <= 10:
        raise ValueError("Golden regression dataset must contain between 5 and 10 cases.")
    parsed: list[GoldenCase] = []
    case_ids: set[str] = set()
    for raw in cases:
        required = {
            "case_id", "source", "question_number", "question_text", "model_answer", "student_answer",
            "max_marks", "expected_status", "required_actions", "reference_response", "retrieval_evidence",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Golden case is missing fields: {sorted(missing)}")
        case_id = str(raw["case_id"])
        if case_id in case_ids:
            raise ValueError(f"Golden case IDs must be unique: {case_id}")
        case_ids.add(case_id)
        status = str(raw["expected_status"])
        if status not in {"completed", "manual_review"}:
            raise ValueError(f"Unsupported expected status in {case_id}: {status}")
        expected_marks = raw.get("expected_awarded_marks")
        if status == "completed" and expected_marks is None:
            raise ValueError(f"Completed case {case_id} must declare expected_awarded_marks.")
        if expected_marks is not None and not 0 <= float(expected_marks) <= float(raw["max_marks"]):
            raise ValueError(f"Expected marks are outside bounds for {case_id}.")
        parsed.append(
            GoldenCase(
                case_id=case_id,
                source=str(raw["source"]),
                question_number=str(raw["question_number"]),
                question_text=str(raw["question_text"]),
                model_answer=str(raw["model_answer"]),
                student_answer=str(raw["student_answer"]),
                max_marks=float(raw["max_marks"]),
                expected_status=status,
                expected_awarded_marks=float(expected_marks) if expected_marks is not None else None,
                required_actions=tuple(str(action) for action in raw["required_actions"]),
                reference_response=str(raw["reference_response"]),
                retrieval_evidence=str(raw["retrieval_evidence"]),
                flagged_ocr_text=raw.get("flagged_ocr_text"),
                max_tool_calls=int(raw.get("max_tool_calls", 5)),
            )
        )
    return parsed


def _fixed_retrieval(case: GoldenCase, **kwargs: Any) -> RagContext:
    query = str(kwargs.get("query") or "golden reference")
    return RagContext(
        reference_set_id="week17-golden-refs",
        chunks=[
            RetrievedChunk(
                text=case.retrieval_evidence,
                source=f"golden/{case.case_id}.txt",
                chunk_index=0,
                reference_set_id="week17-golden-refs",
            ),
            RetrievedChunk(
                text=f"Requested retrieval query: {query}",
                source="golden/query-log.txt",
                chunk_index=1,
                reference_set_id="week17-golden-refs",
            ),
        ],
    )


def _candidate_text(case: GoldenCase, state: Any) -> str:
    grade = state.current_grade
    marks = "none" if grade is None else f"{grade.awarded_marks:g}/{grade.max_marks:g}"
    feedback = "none" if grade is None else grade.feedback
    actions = [record.action for record in state.trajectory if record.action != "planner"]
    return (
        f"candidate_status={state.status}\n"
        f"candidate_marks={marks}\n"
        f"candidate_actions={', '.join(actions) or 'none'}\n"
        f"candidate_feedback={feedback}\n"
        f"candidate_clarification={state.clarification_reason or 'none'}"
    )


def _reference_text(case: GoldenCase) -> str:
    marks = "none" if case.expected_awarded_marks is None else f"{case.expected_awarded_marks:g}/{case.max_marks:g}"
    return (
        f"expected_status={case.expected_status}\nexpected_marks={marks}\n"
        f"required_actions={', '.join(case.required_actions)}\nmax_tool_calls={case.max_tool_calls}\n"
        f"approved_reference={case.reference_response}"
    )


def _project_sanity(case: GoldenCase, state: Any) -> tuple[bool, str]:
    """Independent contract check used to inspect judge verdict plausibility."""
    actions = [record.action for record in state.trajectory if record.action != "planner"]
    if state.status != case.expected_status:
        return False, f"status {state.status!r} differs from expected {case.expected_status!r}"
    if not set(case.required_actions).issubset(actions):
        return False, f"missing required actions: {sorted(set(case.required_actions) - set(actions))}"
    if len(actions) > case.max_tool_calls:
        return False, f"used {len(actions)} tool calls; limit is {case.max_tool_calls}"
    if case.expected_awarded_marks is None:
        return state.current_grade is None, "manual-review case must not carry a grade"
    if state.current_grade is None:
        return False, "completed case has no grade"
    if abs(state.current_grade.awarded_marks - case.expected_awarded_marks) > 0.01:
        return False, f"marks {state.current_grade.awarded_marks:g} differ from expected {case.expected_awarded_marks:g}"
    return True, "status, required actions, tool-call limit, and expected marks match the approved case contract"


def _judge_descriptors(model: str) -> list[Any]:
    shared = [LLMMessage.system("You are a strict Week 16 grading-regression judge. Use only the supplied text.")]
    correctness = BinaryClassificationPromptTemplate(
        criteria=(
            "Classify PASS only if the candidate preserves the approved reference decision: no material "
            "contradiction, omitted required outcome, or unjustified score/status. Treat uncertainty as FAIL."
        ),
        target_category="PASS", non_target_category="FAIL", uncertainty=Uncertainty.NON_TARGET,
        include_reasoning=True, pre_messages=shared,
    )
    workflow = BinaryClassificationPromptTemplate(
        criteria=(
            "Classify PASS only if the candidate follows the approved grading workflow: expected terminal status, "
            "required actions, score policy, and maximum tool-call limit. Treat uncertainty as FAIL."
        ),
        target_category="PASS", non_target_category="FAIL", uncertainty=Uncertainty.NON_TARGET,
        include_reasoning=True, pre_messages=shared,
    )
    return [
        LLMEval("candidate", provider="ollama", model=model, template=correctness,
                additional_columns={"reference": "reference"}, alias="reference_correctness"),
        LLMEval("candidate", provider="ollama", model=model, template=workflow,
                additional_columns={"reference": "reference"}, alias="workflow_adherence"),
    ]


def _run_evidently_judges(rows: list[dict[str, Any]], model: str, report_path: Path) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(
        frame,
        descriptors=_judge_descriptors(model),
        options=[OllamaOptions(api_url="http://localhost:11434")],
        metadata={"judge_provider": "ollama", "judge_model": model},
        tags=["week17", "regression", "llm-judge"],
    )
    judged = dataset.as_dataframe().copy()
    judged["reference_correctness_pass"] = (judged["reference_correctness"] == "PASS").astype(float)
    judged["workflow_adherence_pass"] = (judged["workflow_adherence"] == "PASS").astype(float)
    report = Report([
        RowCount(), MeanValue(column="reference_correctness_pass"), MeanValue(column="workflow_adherence_pass"),
        CategoryCount(column="reference_correctness", categories=["PASS", "FAIL"]),
        CategoryCount(column="workflow_adherence", categories=["PASS", "FAIL"]),
    ], metadata={"judge_provider": "ollama", "judge_model": model}, tags=["week17", "regression"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.run(Dataset.from_pandas(judged)).save_html(str(report_path))
    return judged.to_dict(orient="records")


def run_regression(
    config_version: str = "v3", *, tracking_uri: str | None = None,
    experiment_name: str = REGRESSION_EXPERIMENT_NAME,
) -> RegressionRunResult:
    config = load_experiment_config(config_version)
    settings = get_settings()
    if settings.llm_provider.lower().strip() != "ollama":
        raise ValueError("Week 17 regression requires LLM_PROVIDER=ollama for the configured local agent and judge.")
    cases = load_golden_cases()
    tracking_uri, experiment_id = configure_local_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        provider = get_llm_provider(
            ollama_planner_prompt_template=config.planner_prompt_text(),
            ollama_system_instruction=config.prompt_text(),
        )
        state = run_question_agent(
            question_number=case.question_number, question_text=case.question_text, max_marks=case.max_marks,
            model_answer=case.model_answer, mapped_student_answer=case.student_answer,
            flagged_ocr_text=case.flagged_ocr_text, reference_set_id="week17-golden-refs", planner=provider,
            grade_fn=provider.grade_answer, retrieve_fn=lambda **kwargs: _fixed_retrieval(case, **kwargs),
            max_iterations=max(settings.agent_max_iterations, case.max_tool_calls),
        )
        sanity_pass, sanity_explanation = _project_sanity(case, state)
        candidates.append({
            "case_id": case.case_id, "source": case.source, "candidate": _candidate_text(case, state),
            "reference": _reference_text(case), "project_sanity_pass": sanity_pass,
            "project_sanity_explanation": sanity_explanation, "agent_status": state.status,
            "agent_actions": [record.model_dump() for record in state.trajectory],
            "candidate_awarded_marks": state.current_grade.awarded_marks if state.current_grade else None,
            "expected_awarded_marks": case.expected_awarded_marks,
        })
    report_path = REPORT_DIRECTORY / f"{config.version}_regression.html"
    judged = _run_evidently_judges(candidates, settings.ollama_model, report_path)
    for item in judged:
        item["judge_pass"] = item["reference_correctness"] == "PASS" and item["workflow_adherence"] == "PASS"
        item["passed"] = bool(item["judge_pass"] and item["project_sanity_pass"])
        item["sanity_assessment"] = (
            "Judge and independent project contract agree."
            if item["judge_pass"] == item["project_sanity_pass"]
            else "Judge/project-contract disagreement: preserve for human review; no automatic promotion."
        )
    passed = sum(bool(item["passed"]) for item in judged)
    percentage = 100 * passed / len(judged)
    failures = [str(item["case_id"]) for item in judged if not item["passed"]]
    metrics = {
        "regression_case_count": len(judged), "regression_passed_cases": passed,
        "regression_pass_percentage": percentage, "pct_tests_passed": percentage,
        "judge_correctness_pass_rate": sum(item["reference_correctness"] == "PASS" for item in judged) / len(judged),
        "judge_workflow_pass_rate": sum(item["workflow_adherence"] == "PASS" for item in judged) / len(judged),
        "project_sanity_pass_rate": sum(bool(item["project_sanity_pass"]) for item in judged) / len(judged),
        "regression_latency_seconds": time.perf_counter() - started,
    }
    with mlflow.start_run(run_name=f"{config.version}-fixed-regression") as active_run:
        mlflow.set_tags({"evaluation_type": "fixed_golden_regression", "prompt_version": config.version,
                         "judge_provider": "ollama", "judge_model": settings.ollama_model,
                         "promotion": "interpretation_only"})
        mlflow.log_params({key: str(value) for key, value in resolved_parameters(config, settings).items()})
        mlflow.log_params({"golden_dataset": str(GOLDEN_DATASET_PATH.relative_to(PROJECT_ROOT)), "golden_case_count": len(cases),
                           "judge_checks": "reference_correctness,workflow_adherence", "evidently_version": "0.7.23"})
        mlflow.log_metrics(numeric_metrics(metrics))
        with tempfile.TemporaryDirectory(prefix="week17_regression_") as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "golden_cases.json").write_text(json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8")
            (artifact_dir / "regression_results.json").write_text(json.dumps(judged, indent=2), encoding="utf-8")
            (artifact_dir / "regression_summary.json").write_text(json.dumps({"metrics": metrics, "failures": failures, "report": str(report_path)}, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(artifact_dir))
            mlflow.log_artifact(str(report_path), artifact_path="evidently")
        run_id = active_run.info.run_id
    return RegressionRunResult(run_id, config.version, len(judged), passed, percentage, str(report_path), failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Week 17 golden regression checks.")
    parser.add_argument("--config", default="v3", choices=("v1", "v2", "v3", "all"))
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    versions = ("v1", "v2", "v3") if args.config == "all" else (args.config,)
    for version in versions:
        result = run_regression(version, tracking_uri=args.tracking_uri)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
