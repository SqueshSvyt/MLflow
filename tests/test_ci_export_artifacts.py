"""Перевірка артефактів після src/ci_export_rf.py (інтеграція CI)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pytest


@pytest.mark.integration
def test_ci_export_produces_bundle(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parent.parent
    prepared = root / "data" / "prepared"
    if not (prepared / "train.parquet").exists():
        pytest.skip("data/prepared missing")

    out = tmp_path / "bundle"
    monkeypatch.setenv("CI_ARTIFACT_DIR", str(out))
    subprocess.run(
        [sys.executable, str(root / "src" / "ci_export_rf.py")],
        cwd=str(root),
        check=True,
        env={**os.environ, "CI_ARTIFACT_DIR": str(out), "PYTHONPATH": f"{root}:{root / 'src'}"},
    )

    pkl = out / "model.pkl"
    assert pkl.exists()
    blob = joblib.load(pkl)
    assert "model" in blob and hasattr(blob["model"], "predict")

    metrics_path = out / "metrics.json"
    assert metrics_path.exists()
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert m["test_f1_weighted"] >= 0.0

    png = out / "confusion_matrix.png"
    assert png.exists()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
