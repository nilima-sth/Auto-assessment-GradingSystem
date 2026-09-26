from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from backend.app.mlops.config import PROJECT_ROOT


DEFAULT_EXPERIMENT_NAME = "week16-agentic-grading-system"


def configure_local_tracking(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> tuple[str, str]:
    """Configure local SQLite metadata and local artifacts unless overridden.

    MLflow 3 disables the legacy file-backed tracking store by default. SQLite is
    therefore the local metadata backend, while `mlruns/` remains the local artifact
    directory familiar to MLflow users.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = _local_artifact_location(uri)
        if artifact_location:
            experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
            experiment = client.get_experiment(experiment_id)
        else:
            experiment = mlflow.set_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return uri, experiment.experiment_id


def _local_artifact_location(tracking_uri: str) -> str | None:
    """Keep SQLite-backed local experiment artifacts beside their database."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    database_path = Path(tracking_uri.removeprefix(prefix)).resolve()
    return (database_path.parent / "mlruns").as_uri()


def numeric_metrics(values: dict) -> dict[str, float]:
    """MLflow accepts numeric values only; omit unavailable Week 16 measurements."""
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
