#!/usr/bin/env python3
"""
Перевірка DAG: DagBag без import_errors (для CI / локально).
Потрібен встановлений apache-airflow (версія як у airflow/Dockerfile).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DAG_FOLDER = REPO / "airflow" / "dags"


def main() -> int:
    os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_dag_ci")
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
    os.environ.setdefault("AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION", "true")
    # DAG імпортує лише стандартні пакети + airflow на рівні модуля
    sys.path.insert(0, str(REPO))

    if not DAG_FOLDER.is_dir():
        print(f"Missing DAG folder: {DAG_FOLDER}", file=sys.stderr)
        return 1

    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)
    if bag.import_errors:
        for path, err in sorted(bag.import_errors.items()):
            print(f"{path}\n{err}", file=sys.stderr)
        return 1

    if not bag.dags:
        print("No DAGs parsed.", file=sys.stderr)
        return 1

    print(f"OK: {len(bag.dags)} DAG(s): {', '.join(sorted(bag.dags))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
