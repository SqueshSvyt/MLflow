"""
DVC: навчити всі моделі з data/prepared і зберегти в data/models/<ModelName>/.
Виклик: python src/run_all_models.py data/prepared data/models
Записує metrics.json з метриками по кожній моделі.
"""
import json
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Запуск з кореня проєкту: python src/run_all_models.py ...
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY


def main():
    prepared_dir = pathlib.Path(sys.argv[1]).resolve()
    models_dir = pathlib.Path(sys.argv[2]).resolve()

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

    y_all = pd.concat([y_train, y_test])
    le_dx = LabelEncoder()
    le_dx.fit(y_all)

    all_metrics = {}
    for name in ["RandomForest", "GradientBoosting", "CNN", "ResNet"]:
        module = MODEL_REGISTRY[name]
        model_dir = models_dir / name
        print(f"Training {name} -> {model_dir}")
        try:
            if name in ("CNN", "ResNet"):
                metrics = module.run_dvc(
                    X_train, X_test, y_train, y_test,
                    model_dir=model_dir,
                    le_dx=le_dx,
                    epochs=10,
                    batch_size=64,
                    learning_rate=1e-3,
                )
            else:
                # GradientBoosting повільний: 100 дерев по черзі, 787 ознак. Для DVC зменшуємо.
                metrics = module.run_dvc(
                    X_train, X_test, y_train, y_test,
                    model_dir=model_dir,
                    scaler=scaler,
                    max_depth=10 if name == "RandomForest" else 3,
                    n_estimators=50 if name == "GradientBoosting" else 100,
                    learning_rate=0.1,
                    random_state=42,
                )
            all_metrics[name] = metrics
            print(f"  test_accuracy={metrics['test_accuracy']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics[name] = {"error": str(e)}

    metrics_path = ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
