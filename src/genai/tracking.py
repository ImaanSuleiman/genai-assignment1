"""Thin MLflow wrapper. Tracking never crashes a training run.

Set TRACKER=none to disable. Runs are stored in ./mlflow.db (SQLite) and ./mlartifacts.
View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import os

from .common import ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class Tracker:
    def __init__(self, experiment: str, run_name: str, nested: bool = False, tags: dict | None = None):
        self.enabled = mlflow is not None and os.environ.get("TRACKER", "mlflow") != "none"
        self.run = None
        if not self.enabled:
            return
        try:
            mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlartifacts" / experiment).as_uri())
            mlflow.set_experiment(experiment)
            self.run = mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
        except Exception as e:  # pragma: no cover
            print(f"[tracking] disabled: {e}")
            self.enabled = False

    def params(self, d: dict) -> None:
        if not self.enabled:
            return
        try:
            mlflow.log_params({k: (str(v)[:240]) for k, v in d.items()})
        except Exception as e:
            print(f"[tracking] params: {e}")

    def metrics(self, d: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)
        except Exception as e:
            print(f"[tracking] metrics: {e}")

    def artifact(self, path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            mlflow.log_artifact(str(path), artifact_path)
        except Exception as e:
            print(f"[tracking] artifact: {e}")

    def end(self) -> None:
        if self.enabled and self.run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.end()
