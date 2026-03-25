# Models

Each model has its own folder with a `train.py` that implements `run()`.

| Folder              | Model            | Framework  |
|---------------------|------------------|------------|
| `random_forest/`    | RandomForest     | sklearn    |
| `gradient_boosting/`| GradientBoosting | sklearn    |
| `cnn/`              | SimpleCNN        | PyTorch    |
| `resnet/`           | ResNet18WithMeta | PyTorch    |

- **common.py** — shared plotting (`plot_confusion_matrix`, `plot_feature_importance`) and constants (`IMG_FEATURES`, `META_FEATURES`).
- **Entry point** — `src/train.py` loads data, then calls `MODEL_REGISTRY[args.model].run(...)`.

To add a new model: create `src/models/<name>/` with `__init__.py` and `train.py` (exposing `run`), and register it in `src/models/__init__.py`.
