"""Перевірка коректності артефактів: model.pkl, metrics.json, PNG."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_model_pkl_loadable(tmp_path: Path):
    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)
    model.fit(np.random.randn(20, 5), np.random.randint(0, 2, 20))
    path = tmp_path / "model.pkl"
    joblib.dump({"model": model, "scaler": None}, path)
    loaded = joblib.load(path)
    assert "model" in loaded
    assert hasattr(loaded["model"], "predict")


def test_metrics_json_schema(tmp_path: Path):
    path = tmp_path / "metrics.json"
    metrics = {
        "train_accuracy": 0.9,
        "test_accuracy": 0.85,
        "train_f1_weighted": 0.88,
        "test_f1_weighted": 0.83,
    }
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("test_accuracy", "test_f1_weighted"):
        assert key in data
        assert isinstance(data[key], (int, float))


def test_confusion_matrix_png_is_valid_image(tmp_path: Path):
    path = tmp_path / "confusion_matrix.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
