# Apache Airflow — локальний оркестратор ML-пайплайна

## Що всередині

- **docker-compose.yaml** — Postgres, `airflow-init`, webserver (8080), scheduler.
- Образ на базі **apache/airflow** + **git**, **DVC**, **MLflow** та залежності з кореневого `requirements-docker.txt`.
- Том **`..` → `/opt/airflow/ml_repo`** — ваш ML-код, `data/`, `.dvc`, `mlruns` (read/write для DVC і train).

## Запуск

З **кореня репозиторію** (там, де `requirements-docker.txt`):

```bash
echo "AIRFLOW_UID=$(id -u)" > airflow/.env
docker compose -f airflow/docker-compose.yaml build
docker compose -f airflow/docker-compose.yaml up airflow-init
docker compose -f airflow/docker-compose.yaml up -d
```

Відкрити **http://localhost:8080** — логін `admin` / `admin`.

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
