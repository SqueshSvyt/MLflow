"""
Скрипт навчання моделей з логуванням у MLflow.
Датасет: HAM10000 (зображення hmnist 28x28 + метадані).
Кожна модель в окремій папці в src/models/: random_forest, gradient_boosting, cnn, resnet.
"""
import argparse
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

from data import load_combined, get_raw_data_paths


def run_dvc_train(prepared_dir: pathlib.Path, model_dir: pathlib.Path):
    """DVC mode: load prepared, train RF, save model.pkl and metrics.json."""
    import json
    train_df = pd.read_parquet(prepared_dir / "train.parquet")
    test_df = pd.read_parquet(prepared_dir / "test.parquet")
    target = "dx"
    feature_cols = [c for c in train_df.columns if c != target]
    X_train = train_df[feature_cols].values
    y_train = train_df[target]
    X_test = test_df[feature_cols].values
    y_test = test_df[target]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, y_pred_train))
    test_accuracy = float(accuracy_score(y_test, y_pred_test))
    train_f1 = float(f1_score(y_train, y_pred_train, average="weighted"))
    test_f1 = float(f1_score(y_test, y_pred_test, average="weighted"))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_dir / "model.pkl")
    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1_weighted": train_f1,
        "test_f1_weighted": test_f1,
    }
    metrics_path = pathlib.Path("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"DVC train: test_accuracy={test_accuracy:.4f}, test_f1={test_f1:.4f}")
    print(f"Model saved to {model_dir / 'model.pkl'}, metrics to {metrics_path}")


def main():
    """Entry point. DVC mode: train.py PREPARED_DIR MODEL_DIR. Else: argparse + MODEL_REGISTRY."""
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-"):
        prep = pathlib.Path(sys.argv[1]).resolve()
        if prep.is_dir() and (prep / "train.parquet").exists():
            run_dvc_train(prep, pathlib.Path(sys.argv[2]).resolve())
            return 0

    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models import MODEL_REGISTRY  # noqa: E402

    parser = argparse.ArgumentParser(description="Train model with MLflow")
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "GradientBoosting", "CNN", "ResNet"],
        help="Model type",
    )
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--author", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="v1")
    args = parser.parse_args()

    meta_path, hmnist_path = get_raw_data_paths(root)
    X, y, feature_names = load_combined(meta_path, hmnist_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    module = MODEL_REGISTRY[args.model]
    if args.model in ("CNN", "ResNet"):
        le_dx = LabelEncoder()
        le_dx.fit(y)
        module.run(args, root, X_train, X_test, y_train, y_test, le_dx)
    else:
        module.run(X_train, X_test, y_train, y_test, feature_names, args, root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
