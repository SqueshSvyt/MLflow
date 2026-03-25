# Airflow + DVC + MLflow + залежності ML-проєкту (build з кореня репозиторію).
# docker compose -f airflow/docker-compose.yaml build
FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements-docker.txt /requirements-docker.txt
RUN pip install --no-cache-dir -r /requirements-docker.txt
