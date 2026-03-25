# Multi-stage: важкі залежності збираються в builder, runtime — python:slim.
# Збірка: docker build -t mlflow-ham10000 .
# З опційним PyTorch (CPU): docker build --build-arg INSTALL_TORCH=1 -t mlflow-ham10000:torch .
#
# Запуск (приклад): docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/mlruns:/app/mlruns" mlflow-ham10000 \
#   python src/train.py --prepared data/prepared

ARG PYTHON_VERSION=3.11

# --- Етап 1: збірка wheels (повний образ — компіляція / великі пакети) ---
FROM python:${PYTHON_VERSION}-bookworm AS builder

ARG INSTALL_TORCH=0

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements-docker.txt .

# Спочатку лише requirements-docker; опційно torch+torchvision (CPU, офіційний індекс)
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements-docker.txt
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip wheel --no-cache-dir --wheel-dir=/wheels \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

# --- Етап 2: runtime (мінімальний) ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG INSTALL_TORCH=0

# git — для DVC; libgomp1 — часто потрібен для numpy/sklearn
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /wheels /wheels
COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-docker.txt \
    && if [ "$INSTALL_TORCH" = "1" ]; then \
         pip install --no-cache-dir --no-index --find-links=/wheels torch torchvision; \
       fi \
    && rm -rf /wheels

COPY . .

# MLflow за замовчуванням у /app/mlruns всередині контейнера
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# За замовчуванням — довідка train (перевизнач: CMD у docker run)
CMD ["python", "src/train.py", "--help"]
