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


def _ensure_repo_root_first_on_syspath(root: pathlib.Path) -> None:
    """При `python src/train.py` перший шлях — каталог src; потрібен корінь репо для `import src.models`."""
    root_s = str(root.resolve())
    try:
        while root_s in sys.path:
            sys.path.remove(root_s)
    except ValueError:
        pass
    sys.path.insert(0, root_s)


def _run_training(root, args, X_train, X_test, y_train, y_test, feature_names):
    """Спільна логіка навчання: масштабування + виклик module.run()."""
    _ensure_repo_root_first_on_syspath(root)
    from src.models import MODEL_REGISTRY  # noqa: E402

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


def run_from_config(cfg: dict) -> int:
    """
    Запуск навчання з конфігу (Hydra: config + config/model/*, config/hpo/*).
    Використовує cfg.model (якщо є) та cfg.train для параметрів.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    _ensure_repo_root_first_on_syspath(root)
    paths = cfg.get("paths", {})
    prepared_dir = pathlib.Path(paths.get("prepared_dir", "data/prepared"))
    if not prepared_dir.is_absolute():
        prepared_dir = (root / prepared_dir).resolve()
    repro = cfg.get("reproducibility", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    class Args:
        pass

    args = Args()
    args.prepared = prepared_dir
    args.model = model_cfg.get("name", train_cfg.get("model", "RandomForest"))
    # Tree-based (RandomForest, GradientBoosting)
    args.max_depth = model_cfg.get("max_depth", train_cfg.get("max_depth", 5))
    args.n_estimators = model_cfg.get("n_estimators", train_cfg.get("n_estimators", 100))
    args.learning_rate = model_cfg.get("learning_rate", train_cfg.get("learning_rate", 0.1))
    # CNN / ResNet (із model/*.yaml або train)
    args.epochs = model_cfg.get("epochs", train_cfg.get("epochs", 10))
    args.batch_size = model_cfg.get("batch_size", train_cfg.get("batch_size", 64))
    args.random_state = repro.get("random_state", 42)
    args.author = repro.get("author", "default")
    args.dataset_version = repro.get("dataset_version", "v1")

    params_path = prepared_dir / "params.json"
    if params_path.exists():
        prep_params = json.loads(params_path.read_text(encoding="utf-8"))
        args.test_size = prep_params["test_size"]
        args.split_random_state = prep_params["random_state"]
    else:
        args.test_size = None
        args.split_random_state = None

    X_train, X_test, y_train, y_test, feature_names = load_prepared(prepared_dir)
    _run_training(root, args, X_train, X_test, y_train, y_test, feature_names)
    return 0


def main():
    """Entry point. argparse + MODEL_REGISTRY. Uses data/prepared (prepare.py output) by default."""
    root = pathlib.Path(__file__).resolve().parent.parent
    _ensure_repo_root_first_on_syspath(root)

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
    params_path = prepared_dir / "params.json"
    if params_path.exists():
        prep_params = json.loads(params_path.read_text(encoding="utf-8"))
        args.test_size = prep_params["test_size"]
        args.split_random_state = prep_params["random_state"]
    else:
        args.test_size = None
        args.split_random_state = None

    _run_training(root, args, X_train, X_test, y_train, y_test, feature_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
