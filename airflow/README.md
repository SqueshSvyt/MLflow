# Apache Airflow — локальний оркестратор ML-пайплайна

## Що всередині

- **docker-compose.yaml** — Postgres, `airflow-init`, webserver (8080), scheduler.
- Образ на базі **apache/airflow** + **git**, **DVC**, **MLflow** та залежності з кореневого `requirements-docker.txt`.
- Том **`..` → `/opt/airflow/ml_repo`** — ваш ML-код, `data/`, `.dvc`, `mlruns` (read/write для DVC і train).

## Запуск

З **кореня репозиторію** (там, де `requirements-docker.txt` і каталог `airflow/`). Якщо запускати з `airflow/`, шляхи до томів і `build context` зламаються.

```bash
echo "AIRFLOW_UID=$(id -u)" > airflow/.env
docker compose -f airflow/docker-compose.yaml up -d --build
```

`airflow-init` (міграції БД, користувач `admin`) виконається автоматично перед webserver/scheduler завдяки `depends_on`.

Якщо робили лише `up airflow-init` без `up -d` — інтерфейс **не** з’явиться: init контейнер завершується, а webserver треба підняти окремо командою вище.

Відкрити **http://localhost:8080** — логін `admin` / `admin`.

### Не відкривається 8080

- `docker compose -f airflow/docker-compose.yaml ps` — мають бути `Up` для `postgres`, `airflow-webserver`, `airflow-scheduler`.
- `docker compose -f airflow/docker-compose.yaml logs airflow-webserver --tail 80`
- Перевірте, чи зайнятий порт: `lsof -i :8080` (інший процес або старий контейнер).

Увімкнути DAG **`ml_training_pipeline`** і запустити (Trigger DAG). Переконайтеся, що в `data/raw/` є `HAM10000_metadata.csv` та `hmnist_28_28_L.csv`, і що виконано `dvc init` (є каталог `.dvc`).

## Поріг F1 (гілка в DAG)

**Admin → Variables** — додати `F1_THRESHOLD` (число, наприклад `0.35`). Якщо змінної немає, використовується `0.25`.

## Альтернатива: DockerOperator

Зараз кроки виконуються **в контейнері Airflow** з монтуванням репо. Можна замінити `BashOperator` на **`DockerOperator`**, який запускає ваш образ з кореневого `Dockerfile` і монтує ті самі дані — логіка DAG лишається схожою.

## Зупинка

```bash
docker compose -f airflow/docker-compose.yaml down
```

Дані Postgres зберігаються у volume `postgres-db-volume`.

## Якщо впав один з кроків DAG

У UI: **DAG → Graph** — червоний таск → **Log**. Нижче типові причини за `task_id`:

| Таск | Що робить | Часті причини |
|------|-----------|----------------|
| `wait_for_raw_data` | Перевіряє CSV у `data/raw/` | Немає `HAM10000_metadata.csv` / `hmnist_28_28_L.csv` у змонтованому репо |
| `check_dvc_repo` | `dvc status` | Немає `.dvc`; помилка remote (MinIO); битий lock; том примонтовано read-only |
| `prepare_data` | `dvc repro -s prepare` | Помилка `prepare.py`; немає прав на запис у `data/prepared` (налаштуйте `AIRFLOW_UID` у `airflow/.env`) |
| `train_model` | `python src/train.py ...` | Немає `data/prepared/*.parquet`; зламаний `PYTHONPATH`; помилка в коді / залежностях |
| `evaluate_model` | Читає останній run у MLflow | Немає експерименту / пошкоджений `mlruns` (зайві порожні `mlruns/<id>/`) |
| `register_mlflow_staging` | Model Registry | Немає `run_id`; немає артефакту `model` або `model.pkl`; обмеження file-store — у логу тепер детальніший текст помилки |

Після **`branch_on_f1`** один з наступних тасків буде **skipped** (не failed) — це нормально: або реєстрація, або `notify_below_threshold`.
