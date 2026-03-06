"""
Скрипт навчання моделей з логуванням у MLflow.
Датасет: HAM10000 (зображення hmnist 28x28 + метадані).
Кожна модель в окремій папці в src/models/: random_forest, gradient_boosting, cnn, resnet.
"""
import argparse
import json
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler



def load_prepared(prepared_dir: pathlib.Path):
    """Load train/test from prepare.py output (train.parquet, test.parquet)."""
    train_path = prepared_dir / "train.parquet"
    test_path = prepared_dir / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found in {prepared_dir}. "
            "Run: python src/prepare.py data/raw data/prepared"
        )
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    target = "dx"
    feature_cols = [c for c in train_df.columns if c != target]
    X_train = train_df[feature_cols].values
    y_train = train_df[target]
    X_test = test_df[feature_cols].values
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test, feature_cols


def main():
    """Entry point. argparse + MODEL_REGISTRY. Uses data/prepared (prepare.py output) by default."""
    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models import MODEL_REGISTRY  # noqa: E402

    parser = argparse.ArgumentParser(description="Train model with MLflow")
    parser.add_argument(
        "--prepared",
        type=pathlib.Path,
        default=None,
        help="Path to prepared data (train.parquet, test.parquet). Default: data/prepared",
    )
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
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--author", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="v1")
    args = parser.parse_args()

    prepared_dir = (args.prepared or root / "data" / "prepared").resolve()
    X_train, X_test, y_train, y_test, feature_names = load_prepared(prepared_dir)
    # Use prepare step's split params for MLflow logging
    params_path = prepared_dir / "params.json"
    if params_path.exists():
        prep_params = json.loads(params_path.read_text(encoding="utf-8"))
        args.test_size = prep_params["test_size"]
        args.split_random_state = prep_params["random_state"]
    else:
        args.test_size = None
        args.split_random_state = None
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    module = MODEL_REGISTRY[args.model]
    if args.model in ("CNN", "ResNet"):
        le_dx = LabelEncoder()
        le_dx.fit(pd.concat([y_train, y_test]).astype(str))
        module.run(args, root, X_train, X_test, y_train, y_test, le_dx)
    else:
        module.run(X_train, X_test, y_train, y_test, feature_names, args, root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
