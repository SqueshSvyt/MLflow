"""
Експорт для CI/CD: model.pkl, metrics.json, confusion_matrix.png (без MLflow UI).
Читає data/prepared, RF, вивід у CI_ARTIFACT_DIR (default: ci_artifacts).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
# common без models/__init__ (не тягнемо torch у CI)
_common_path = ROOT / "src" / "models" / "common.py"
_spec = importlib.util.spec_from_file_location("models_common_standalone", _common_path)
_common = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_common)
plot_confusion_matrix = _common.plot_confusion_matrix


def main() -> int:
    out = pathlib.Path(os.environ.get("CI_ARTIFACT_DIR", "ci_artifacts")).resolve()
    prepared = ROOT / "data" / "prepared"
    train_df = pd.read_parquet(prepared / "train.parquet")
    test_df = pd.read_parquet(prepared / "test.parquet")
    target = "dx"
    feat = [c for c in train_df.columns if c != target]
    X_train = train_df[feat].values
    y_train = train_df[target]
    X_test = test_df[feat].values
    y_test = test_df[target]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(max_depth=6, n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "train_f1_weighted": float(f1_score(y_train, model.predict(X_train), average="weighted")),
        "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, out / "model.pkl")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_path = out / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path, "RandomForest_CI")
    print(f"Exported to {out}: model.pkl, metrics.json, confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
