"""
DAG: підготовка даних → train → оцінка F1 → гілка → MLflow Registry (Staging) або сповіщення.

Залежить від змінних середовища в Docker Compose:
  ML_REPO_PATH — корінь репозиторію (змонтований том)
  MLFLOW_TRACKING_URI — зазвичай file:/opt/airflow/ml_repo/mlruns

Airflow Variable (опційно):
  F1_THRESHOLD — мінімальний test_f1_weighted для переходу до реєстрації (default 0.25)
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

ML_REPO = os.environ.get("ML_REPO_PATH", "/opt/airflow/ml_repo")
EXPERIMENT = "ham10000_baseline"
REGISTERED_MODEL_NAME = "ham10000_airflow"


def _op_env() -> dict:
    return {
        **os.environ,
        "ML_REPO_PATH": ML_REPO,
        "PYTHONPATH": f"{ML_REPO}:{ML_REPO}/src",
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", f"file:{ML_REPO}/mlruns"),
    }


def wait_for_raw_data(**_context) -> None:
    """Перевірка наявності сирих файлів (аналог sensor)."""
    raw = Path(ML_REPO) / "data" / "raw"
    required = ("HAM10000_metadata.csv", "hmnist_28_28_L.csv")
    missing = [f for f in required if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні файли в {raw}: {missing}")


def check_dvc_repo(**_context) -> None:
    """Перевірка, що DVC ініціалізований у змонтованому репозиторії."""
    repo = Path(ML_REPO)
    if not (repo / ".dvc").exists():
        raise FileNotFoundError(f"Немає {repo}/.dvc — виконайте `dvc init` у корені репозиторію.")
    proc = subprocess.run(
        ["dvc", "status"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dvc status завершився з кодом {proc.returncode}: {proc.stderr or proc.stdout}. "
            "Перевірте з хоста: dvc status; remote MinIO; lock .dvc/tmp; чи не змонтовано репо лише для читання."
        )


def evaluate_latest_run(**context) -> None:
    """Читає останній (не-HPO) run і пушить метрики в XCom."""
    import mlflow
    from mlflow.tracking import MlflowClient

    log = logging.getLogger(__name__)
    tracking = os.environ.get("MLFLOW_TRACKING_URI", f"file:{ML_REPO}/mlruns")
    mlflow.set_tracking_uri(tracking)
    client = MlflowClient(tracking_uri=tracking)
    ti = context["ti"]

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        ti.xcom_push(key="test_f1", value=0.0)
        ti.xcom_push(key="run_id", value="")
        ti.xcom_push(key="model_uri_suffix", value="model")
        return

    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
    except Exception as exc:
        log.warning("evaluate_model: search_runs не вдалося (%s) — перевірте цілісність %s", exc, tracking)
        ti.xcom_push(key="test_f1", value=0.0)
        ti.xcom_push(key="run_id", value="")
        ti.xcom_push(key="model_uri_suffix", value="model")
        return
    runs = [r for r in runs if r.data.tags.get("run_type") != "hpo_study"] or runs
    if not runs:
        ti.xcom_push(key="test_f1", value=0.0)
        ti.xcom_push(key="run_id", value="")
        ti.xcom_push(key="model_uri_suffix", value="model")
        return

    run = runs[0]
    rid = run.info.run_id
    metrics = run.data.metrics
    f1 = metrics.get("test_f1_weighted") or metrics.get("f1_weighted") or 0.0

    paths: list[str] = []
    to_visit = [""]
    while to_visit:
        prefix = to_visit.pop()
        for item in client.list_artifacts(rid, path=prefix):
            paths.append(item.path)
            if item.is_dir:
                to_visit.append(item.path)

    # MLflow 3 register_model потребує logged model (каталог від sklearn.log_model), не raw model.pkl.
    # Тому пріоритет — завжди "model", .pkl лише як запасний варіант для старих схем.
    suffix = "model"
    if any(p == "model" or p.startswith("model/") for p in paths):
        suffix = "model"
    elif any(p == "model.pkl" or p.endswith("/model.pkl") for p in paths):
        suffix = "model.pkl"

    ti.xcom_push(key="test_f1", value=float(f1))
    ti.xcom_push(key="run_id", value=rid)
    ti.xcom_push(key="model_uri_suffix", value=suffix)


def choose_register_or_notify(**context) -> str:
    """BranchPythonOperator: поріг F1 з Airflow Variable."""
    ti = context["ti"]
    f1 = ti.xcom_pull(key="test_f1", task_ids="evaluate_model")
    threshold = float(Variable.get("F1_THRESHOLD", default_var="0.25"))
    if f1 is not None and float(f1) >= threshold:
        return "register_mlflow_staging"
    return "notify_below_threshold"


def register_model_staging(**context) -> None:
    """Реєстрація моделі в MLflow Model Registry, стадія Staging (якщо підтримується)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="evaluate_model")
    suffix = ti.xcom_pull(key="model_uri_suffix", task_ids="evaluate_model") or "model"
    if not run_id:
        raise ValueError(
            "Немає run_id після train — перевірте MLFLOW_TRACKING_URI, експеримент ham10000_baseline "
            "та логи таску train_model (чи записався run у mlruns на змонтованому томі)."
        )

    tracking = os.environ.get("MLFLOW_TRACKING_URI", f"file:{ML_REPO}/mlruns")
    mlflow.set_tracking_uri(tracking)
    # Спочатку "model" (sklearn MLmodel), потім model.pkl; XCom suffix може бути застарілим після зміни evaluate.
    try_order: list[str] = []
    for s in ("model", "model.pkl", suffix):
        if s and s not in try_order:
            try_order.append(s)

    mv = None
    last_exc: BaseException | None = None
    for art in try_order:
        uri = f"runs:/{run_id}/{art}"
        try:
            mv = mlflow.register_model(uri, REGISTERED_MODEL_NAME)
            break
        except Exception as exc:
            last_exc = exc
    if mv is None:
        raise RuntimeError(
            f"register_model не вдалося для run {run_id}; спробовано {try_order}. "
            f"Остання помилка: {last_exc!r}. URI={tracking}."
        ) from last_exc

    client = MlflowClient(tracking_uri=tracking)
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=int(mv.version),
            stage="Staging",
            archive_existing_versions=False,
        )
    except Exception:
        # Новіші версії MLflow можуть змінювати API стадій
        pass


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="DVC prepare → train → F1 gate → MLflow Registry (Staging)",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "dvc", "mlflow"],
    doc_md=__doc__,
) as dag:
    sensor_data = PythonOperator(
        task_id="wait_for_raw_data",
        python_callable=wait_for_raw_data,
    )

    dvc_check = PythonOperator(
        task_id="check_dvc_repo",
        python_callable=check_dvc_repo,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command='set -euo pipefail; cd "$ML_REPO_PATH" && dvc repro -s prepare',
        env=_op_env(),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            'set -euo pipefail; cd "$ML_REPO_PATH" && '
            "python src/train.py --prepared data/prepared --model RandomForest "
            "--max_depth 5 --n_estimators 50 --random_state 42"
        ),
        env=_op_env(),
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_latest_run,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_f1",
        python_callable=choose_register_or_notify,
    )

    register_mlflow_staging = PythonOperator(
        task_id="register_mlflow_staging",
        python_callable=register_model_staging,
    )

    notify_below_threshold = BashOperator(
        task_id="notify_below_threshold",
        bash_command=(
            'echo "F1 нижче порога F1_THRESHOLD — реєстрацію пропущено. '
            'Перевірте метрики в MLflow UI."'
        ),
        env=_op_env(),
    )

    sensor_data >> dvc_check >> prepare_data >> train_model >> evaluate_model >> branch
    branch >> register_mlflow_staging
    branch >> notify_below_threshold
