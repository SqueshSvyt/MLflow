"""
Quality Gate за метриками після train (MLflow).
У CI запускати після: prepare + train; MLFLOW_TRACKING_URI за замовчуванням ./mlruns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

EXPERIMENT = "ham10000_baseline"


@pytest.mark.integration
def test_latest_run_f1_above_threshold():
    root = Path(__file__).resolve().parent.parent
    tracking = root / "mlruns"
    if not tracking.exists():
        pytest.skip("mlruns missing — run train first")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(str(tracking))
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        pytest.skip(f"experiment {EXPERIMENT} not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=15,
    )
    assert runs, "no runs in experiment"
    runs = [r for r in runs if r.data.tags.get("run_type") != "hpo_study"] or runs

    f1 = runs[0].data.metrics.get("test_f1_weighted") or runs[0].data.metrics.get("f1_weighted")
    assert f1 is not None, "test_f1_weighted not logged"
    min_f1 = float(os.environ.get("QUALITY_F1_MIN", "0.05"))
    assert float(f1) >= min_f1, f"Quality gate: test_f1_weighted={f1} < {min_f1}"


@pytest.mark.integration
def test_latest_run_has_model_artifact():
    root = Path(__file__).resolve().parent.parent
    tracking = root / "mlruns"
    if not tracking.exists():
        pytest.skip("mlruns missing")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=str(tracking))
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        pytest.skip("no experiment")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=15,
    )
    if not runs:
        pytest.skip("no runs")
    runs = [r for r in runs if r.data.tags.get("run_type") != "hpo_study"] or runs

    run = runs[0]
    rid = run.info.run_id

    # MLflow 3 може зберігати модель не лише як артефакт "model/",
    # тому перевіряємо і tag історії log_model, і файлові артефакти.
    history = run.data.tags.get("mlflow.log-model.history")
    if history:
        return

    to_visit = [""]
    paths: list[str] = []
    while to_visit:
        prefix = to_visit.pop()
        for item in client.list_artifacts(rid, path=prefix):
            paths.append(item.path)
            if item.is_dir:
                to_visit.append(item.path)

    assert any(
        p == "model"
        or p.startswith("model/")
        or p.endswith("/MLmodel")
        or p == "MLmodel"
        for p in paths
    ), paths
