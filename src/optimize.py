"""
HPO + MLflow + Hydra: optuna | random | grid.

Реалізовано:
• Вибір моделі на основі конфігурації — model_cfg.name (config/model/*.yaml), _build_model().
• Оптимізація гіперпараметрів через trial.suggest_* — Optuna: _suggest_param() → suggest_int/suggest_float.
• Коректна оцінка: мінімум train/val split (holdout); опційно cross-validation (evaluation: cv, n_folds).
• Вкладені MLflow runs для кожного trial — log_nested_run() з mlflow.start_run(nested=True).

Запуск:
  python src/optimize.py
  python src/optimize.py model=gradient_boosting hpo=random
  python src/optimize.py hpo.evaluation=cv hpo.n_folds=5
"""
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _git_commit(root: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _log_reproducibility_mlflow(
    root: pathlib.Path,
    *,
    dataset_version: str = "v1",
    random_state: int | None = None,
    config_dict: dict | None = None,
) -> None:
    """Логує в поточний MLflow run теги/параметри для відтворюваності."""
    mlflow.set_tag("dataset_version", dataset_version)
    if random_state is not None:
        mlflow.log_param("random_state", random_state)
    commit = _git_commit(root)
    if commit:
        mlflow.set_tag("code_commit", commit)
    if config_dict:
        mlflow.log_dict(config_dict, "config/reproducibility.json")


def load_prepared(prepared_dir: pathlib.Path):
    """Завантажити train/test з виходу prepare.py."""
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


def _get_cfg():
    """Завантажити конфіг через Hydra (з overrides) або OmegaConf."""
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        config_dir = str(ROOT / "config")
        overrides = sys.argv[1:]
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config", overrides=overrides)
        return OmegaConf.to_container(cfg, resolve=True) or {}
    except Exception:
        pass
    # Fallback: один файл config.yaml без груп model/hpo
    from omegaconf import OmegaConf
    p = ROOT / "config" / "config.yaml"
    if p.exists():
        c = OmegaConf.load(p)
        return OmegaConf.to_container(c, resolve=True) or {}
    return {}


def _build_model(model_name: str, params: dict, random_state: int):
    """Створити екземпляр моделі за назвою та параметрами (sklearn для HPO)."""
    if model_name == "RandomForest":
        return RandomForestClassifier(
            max_depth=params.get("max_depth", 5),
            n_estimators=params.get("n_estimators", 100),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
        )
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            max_depth=params.get("max_depth", 5),
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
        )
    raise ValueError(f"optimize.py підтримує RandomForest, GradientBoosting, CNN; отримано: {model_name}")


def _train_eval_cnn(
    X_train_s: np.ndarray,
    X_val_s: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    le_dx,
    params: dict,
    random_state: int,
) -> tuple[float, float]:
    """Один trial CNN: навчання на train, оцінка на val. Повертає (val_f1, val_acc)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.common import IMG_FEATURES, META_FEATURES
    from src.models.cnn.train import SimpleCNN

    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(le_dx.classes_)
    epochs = int(params.get("epochs", 8))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("learning_rate", 0.001))

    def to_img(X):
        t = torch.from_numpy(X.astype(np.float32))
        return t.reshape(-1, 1, 28, 28)

    X_tr_img = to_img(X_train_s[:, :IMG_FEATURES])
    X_tr_meta = torch.from_numpy(X_train_s[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES].astype(np.float32))
    y_tr = torch.from_numpy(le_dx.transform(y_train)).long()
    train_ds = TensorDataset(X_tr_img, X_tr_meta, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=n_classes, use_meta=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x_img, x_meta, yb in train_loader:
            x_img, x_meta, yb = x_img.to(device), x_meta.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_img, x_meta), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    X_val_img = to_img(X_val_s[:, :IMG_FEATURES])
    X_val_meta = torch.from_numpy(X_val_s[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES].astype(np.float32)).to(device)
    X_val_img = X_val_img.to(device)
    with torch.no_grad():
        y_pred_idx = model(X_val_img, X_val_meta).argmax(dim=1).cpu().numpy()
    y_pred = le_dx.inverse_transform(y_pred_idx)
    f1 = float(f1_score(y_val, y_pred, average="weighted"))
    acc = float((y_pred == y_val).mean())
    return f1, acc


def _suggest_param(trial, key: str, space: dict, default_lo, default_hi, use_float: bool = False):
    """Один параметр: suggest_int або suggest_float з search_space."""
    if key not in space or not isinstance(space.get(key), (list, tuple)) or len(space[key]) < 2:
        if use_float:
            return trial.suggest_float(key, float(default_lo), float(default_hi))
        return trial.suggest_int(key, int(default_lo), int(default_hi))
    lo, hi = space[key][0], space[key][1]
    if use_float or isinstance(lo, float) or isinstance(hi, float):
        return trial.suggest_float(key, float(lo), float(hi))
    return trial.suggest_int(key, int(lo), int(hi))


def _sample_random_trials(model_cfg: dict, n_iter: int, random_state: int) -> list[dict]:
    """Згенерувати n_iter випадкових комбінацій з search_space (межі [low, high])."""
    space = model_cfg.get("search_space") or {}
    rng = np.random.default_rng(random_state)
    trials = []
    for _ in range(n_iter):
        params = {}
        for key, bounds in space.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            lo, hi = bounds[0], bounds[1]
            if isinstance(lo, int) and isinstance(hi, int):
                params[key] = int(rng.integers(lo, hi + 1))
            else:
                params[key] = float(rng.uniform(lo, hi))
        if params:
            trials.append(params)
    return trials if trials else [{}]


def _grid_trials(model_cfg: dict, max_combinations: int | None, random_state: int) -> list[dict]:
    """Згенерувати сітку з search_space. Список з 2 чисел [lo, hi] → кроки; 3+ елементів → дискретні значення."""
    grid_space = model_cfg.get("search_space_grid") or model_cfg.get("search_space") or {}
    axes = {}
    for key, val in grid_space.items():
        if not isinstance(val, (list, tuple)):
            axes[key] = [val]
            continue
        val = list(val)
        if len(val) >= 3:
            axes[key] = [int(x) if isinstance(val[0], int) else float(x) for x in val]
        elif len(val) == 2:
            lo, hi = val[0], val[1]
            if isinstance(lo, int):
                # кілька точок з діапазону
                step = max(1, (hi - lo) // 3)
                axes[key] = list(range(lo, hi + 1, step))[:5]
            else:
                axes[key] = list(np.linspace(lo, hi, 4))
        else:
            axes[key] = [val[0]] if val else [None]
    if not axes:
        return [{}]
    keys = list(axes.keys())
    from itertools import product
    combos = [dict(zip(keys, p)) for p in product(*(axes[k] for k in keys))]
    if max_combinations and len(combos) > max_combinations:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(combos))[:max_combinations]
        combos = [combos[i] for i in perm]
    return combos


def run_optimize(
    root: pathlib.Path,
    prepared_dir: pathlib.Path,
    *,
    model_cfg: dict,
    hpo_cfg: dict,
    repro_cfg: dict,
    model_registry_cfg: dict,
    config_dict: dict | None = None,
) -> dict[str, Any]:
    """Запустити HPO (optuna / random / grid) з nested MLflow runs."""
    X_train_full, X_test, y_train_full, y_test, _ = load_prepared(prepared_dir)
    model_name = model_cfg.get("name", "RandomForest")
    if model_name not in ("RandomForest", "GradientBoosting", "CNN"):
        raise ValueError(
            f"optimize.py HPO підтримує RandomForest, GradientBoosting, CNN; обрано: {model_name}."
        )
    is_cnn = model_name == "CNN"
    if is_cnn:
        evaluation = "holdout"  # CV не підтримується для CNN у цій реалізації
    else:
        evaluation = hpo_cfg.get("evaluation", "holdout")
    val_ratio = float(hpo_cfg.get("val_ratio", 0.2))
    val_random_state = int(hpo_cfg.get("val_random_state", 42))
    random_state = int(repro_cfg.get("random_state", 42))
    metric = hpo_cfg.get("metric", "val_f1_weighted")
    study_name = hpo_cfg.get("study_name", "ham10000_hpo")
    register_best = model_registry_cfg.get("register_best", False)
    stage = model_registry_cfg.get("stage", "Staging")
    n_folds = int(hpo_cfg.get("n_folds", 5))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, random_state=val_random_state, stratify=y_train_full
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_train_full_s = scaler.fit_transform(X_train_full)

    strategy = hpo_cfg.get("strategy", "optuna")
    if is_cnn and strategy != "optuna":
        strategy = "optuna"  # CNN поки тільки з Optuna
    n_trials = int(hpo_cfg.get("n_trials", hpo_cfg.get("n_iter", 20)))
    direction = hpo_cfg.get("direction", "maximize")
    maximize = direction == "maximize"

    le_dx = None
    if is_cnn:
        le_dx = LabelEncoder()
        le_dx.fit(pd.concat([y_train_full, y_test]).astype(str))

    def train_eval(params: dict) -> tuple[float, float]:
        if is_cnn:
            return _train_eval_cnn(X_train_s, X_val_s, y_train, y_val, le_dx, params, random_state)
        model = _build_model(model_name, params, random_state)
        if evaluation == "cv":
            scorer = make_scorer(f1_score, average="weighted")
            scores = cross_val_score(model, X_train_full_s, y_train_full, cv=n_folds, scoring=scorer, n_jobs=1)
            f1 = float(scores.mean())
            acc = f1  # для cv ок логувати той самий показник у val_accuracy
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            f1 = float(f1_score(y_val, y_pred, average="weighted"))
            acc = float((y_pred == y_val).mean())
        return f1, acc

    def log_nested_run(params: dict, score: float, val_accuracy: float):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_type", model_name)
            mlflow.log_params({**params, "random_state": random_state})
            mlflow.log_metric("val_f1_weighted", score)
            mlflow.log_metric("val_accuracy", val_accuracy)

    best_score = -np.inf if maximize else np.inf
    best_params = None

    mlflow.set_experiment("ham10000_baseline")
    with mlflow.start_run(run_name=study_name) as parent_run:
        _log_reproducibility_mlflow(
            root,
            dataset_version=repro_cfg.get("dataset_version", "v1"),
            random_state=random_state,
            config_dict=config_dict,
        )
        mlflow.set_tag("run_type", "hpo_study")
        mlflow.set_tag("hpo_strategy", strategy)
        mlflow.set_tag("author", repro_cfg.get("author", "default"))
        mlflow.log_param("hpo_strategy", strategy)
        mlflow.log_param("hpo_n_trials", n_trials)
        mlflow.log_param("hpo_val_ratio", val_ratio)
        mlflow.log_param("hpo_metric", metric)
        mlflow.log_param("hpo_evaluation", evaluation)
        if evaluation == "cv":
            mlflow.log_param("hpo_n_folds", n_folds)

        if strategy == "optuna":
            import optuna
            space = model_cfg.get("search_space") or {}
            def objective(trial):
                params = {}
                if is_cnn:
                    params["learning_rate"] = _suggest_param(trial, "learning_rate", space, 0.0001, 0.01, use_float=True)
                    params["batch_size"] = _suggest_param(trial, "batch_size", space, 32, 128)
                    params["epochs"] = _suggest_param(trial, "epochs", space, 4, 14)
                else:
                    if "max_depth" in space or model_name in ("RandomForest", "GradientBoosting"):
                        params["max_depth"] = _suggest_param(trial, "max_depth", space, 2, 20)
                    if "n_estimators" in space or model_name in ("RandomForest", "GradientBoosting"):
                        params["n_estimators"] = _suggest_param(trial, "n_estimators", space, 50, 300)
                    if "min_samples_split" in space or model_name in ("RandomForest", "GradientBoosting"):
                        params["min_samples_split"] = _suggest_param(trial, "min_samples_split", space, 2, 20)
                    if "min_samples_leaf" in space or model_name in ("RandomForest", "GradientBoosting"):
                        params["min_samples_leaf"] = _suggest_param(trial, "min_samples_leaf", space, 1, 10)
                    if model_name == "GradientBoosting" and ("learning_rate" in space or True):
                        params["learning_rate"] = _suggest_param(trial, "learning_rate", space, 0.01, 0.3, use_float=True)
                score, val_acc = train_eval(params)
                log_nested_run(params, score, val_acc)
                return score
            study = optuna.create_study(direction=direction, study_name=study_name)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            best_params = study.best_params
            best_score = study.best_value
        else:
            if strategy == "random":
                trials = _sample_random_trials(model_cfg, n_trials, random_state)
            else:
                trials = _grid_trials(model_cfg, hpo_cfg.get("max_combinations"), random_state)
                if len(trials) > n_trials:
                    trials = trials[:n_trials]
            for params in trials:
                if not params:
                    params = {k: model_cfg.get(k) for k in ("max_depth", "n_estimators", "min_samples_split", "min_samples_leaf") if k in model_cfg}
                score, val_acc = train_eval(params)
                log_nested_run(params, score, val_acc)
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()

        if best_params is None:
            best_params = {}
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{metric}", best_score)

        if is_cnn:
            # Фінальна CNN: тренування на повному train, оцінка на test, лог PyTorch + le_dx
            import torch
            import pickle
            import mlflow.pytorch as mlflow_pt
            from torch.utils.data import TensorDataset, DataLoader
            from src.models.common import IMG_FEATURES, META_FEATURES
            from src.models.cnn.train import SimpleCNN

            torch.manual_seed(random_state)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_classes = len(le_dx.classes_)
            epochs = int(best_params.get("epochs", 8))
            batch_size = int(best_params.get("batch_size", 64))
            lr = float(best_params.get("learning_rate", 0.001))

            def to_img(X):
                t = torch.from_numpy(X.astype(np.float32))
                return t.reshape(-1, 1, 28, 28)

            X_full_img = to_img(X_train_full_s[:, :IMG_FEATURES])
            X_full_meta = torch.from_numpy(X_train_full_s[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES].astype(np.float32))
            y_full = torch.from_numpy(le_dx.transform(y_train_full)).long()
            full_ds = TensorDataset(X_full_img, X_full_meta, y_full)
            full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, num_workers=0)

            final_model = SimpleCNN(num_classes=n_classes, use_meta=True).to(device)
            optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(epochs):
                final_model.train()
                for x_img, x_meta, yb in full_loader:
                    x_img, x_meta, yb = x_img.to(device), x_meta.to(device), yb.to(device)
                    optimizer.zero_grad()
                    criterion(final_model(x_img, x_meta), yb).backward()
                    optimizer.step()

            final_model.eval()
            X_te_img = to_img(X_test_s[:, :IMG_FEATURES]).to(device)
            X_te_meta = torch.from_numpy(X_test_s[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES].astype(np.float32)).to(device)
            with torch.no_grad():
                y_test_pred_idx = final_model(X_te_img, X_te_meta).argmax(dim=1).cpu().numpy()
            y_test_pred = le_dx.inverse_transform(y_test_pred_idx)
            test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
            test_accuracy = float((y_test_pred == y_test).mean())
            mlflow.log_metric("final_test_f1_weighted", test_f1)
            mlflow.log_metric("final_test_accuracy", test_accuracy)
            mlflow_pt.log_model(final_model, "best_model")
            le_path = root / "label_encoder_cnn.pkl"
            with open(le_path, "wb") as f:
                pickle.dump(le_dx, f)
            mlflow.log_artifact(str(le_path), "best_model")
            le_path.unlink(missing_ok=True)
        else:
            final_model = _build_model(model_name, best_params, random_state)
            final_model.fit(X_train_full_s, y_train_full)
            y_test_pred = final_model.predict(X_test_s)
            test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
            test_accuracy = float((y_test_pred == y_test).mean())
            mlflow.log_metric("final_test_f1_weighted", test_f1)
            mlflow.log_metric("final_test_accuracy", test_accuracy)
            mlflow.sklearn.log_model(final_model, "best_model")

        mlflow.set_tag("best_model_artifact", "best_model")
        if config_dict:
            mlflow.log_dict(config_dict, "config/final_config.json")

        # (A) За наявності MLflow tracking server з backend store: реєстрація в Registry + Staging
        registered_model_name = model_registry_cfg.get("registered_model_name", "ham10000_best")
        if register_best:
            try:
                model_uri = f"runs:/{parent_run.info.run_id}/best_model"
                result = mlflow.register_model(model_uri, registered_model_name)
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=registered_model_name, version=result.version, stage=stage
                )
                mlflow.log_param("registered_model_name", registered_model_name)
                mlflow.log_param("registered_version", result.version)
                mlflow.log_param("registered_stage", stage)
            except Exception as e:
                mlflow.log_param("register_error", str(e))

        return {
            "best_params": best_params,
            "best_value": best_score,
            "final_test_f1_weighted": test_f1,
            "final_test_accuracy": test_accuracy,
            "run_id": parent_run.info.run_id,
        }


def main():
    root = ROOT
    prepared_dir = root / "data" / "prepared"
    cfg = _get_cfg()
    model_cfg = cfg.get("model") or {}
    hpo_cfg = cfg.get("hpo") or {}
    repro_cfg = cfg.get("reproducibility") or {}
    model_registry_cfg = cfg.get("model_registry") or {}

    result = run_optimize(
        root,
        prepared_dir,
        model_cfg=model_cfg,
        hpo_cfg=hpo_cfg,
        repro_cfg=repro_cfg,
        model_registry_cfg=model_registry_cfg,
        config_dict=cfg,
    )
    print("Best params:", result["best_params"])
    print("Best", hpo_cfg.get("metric", "val_f1_weighted"), "=", result["best_value"])
    print("Final test_f1_weighted:", result["final_test_f1_weighted"])
    print("Final test_accuracy:", result["final_test_accuracy"])
    print("MLflow parent run_id:", result["run_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
