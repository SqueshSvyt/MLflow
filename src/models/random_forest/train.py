"""RandomForest model: train and log to MLflow or run_dvc for DVC pipeline."""

import pathlib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

from ..common import plot_feature_importance, plot_confusion_matrix, log_reproducibility_mlflow


def build_model(args):
    return RandomForestClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


def get_params(args):
    return {"max_depth": args.max_depth, "n_estimators": args.n_estimators}


def run(X_train, X_test, y_train, y_test, feature_names, args, root: pathlib.Path):
    model = build_model(args)
    params = get_params(args)
    mlflow.set_experiment("ham10000_baseline")
    with mlflow.start_run():
        log_reproducibility_mlflow(
            root,
            dataset_version=args.dataset_version,
            random_state=args.random_state,
            test_size=getattr(args, "test_size", None),
            split_random_state=getattr(args, "split_random_state", None),
        )
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("data_source", "images+metadata")
        mlflow.log_param("model", "RandomForest")
        if getattr(args, "test_size", None) is not None:
            mlflow.log_param("test_size", args.test_size)
        if getattr(args, "split_random_state", None) is not None:
            mlflow.log_param("split_random_state", args.split_random_state)
        mlflow.log_param("random_state", args.random_state)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_f1 = f1_score(y_train, y_pred_train, average="weighted")
        test_f1 = f1_score(y_test, y_pred_test, average="weighted")

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_f1_weighted", train_f1)
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.log_metric("accuracy", test_accuracy)
        mlflow.log_metric("f1_weighted", test_f1)

        importance_path = root / "feature_importance.png"
        plot_feature_importance(model, feature_names, importance_path, "RandomForest", top_n=25)
        mlflow.log_artifact(str(importance_path))
        importance_path.unlink(missing_ok=True)

        cm_path = root / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, cm_path, "RandomForest")
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        mlflow.sklearn.log_model(model, "model")
        # Fallback artifact for CI/FS backends where model directory may be absent in run artifacts.
        model_pkl_path = root / "model.pkl"
        joblib.dump(model, model_pkl_path)
        mlflow.log_artifact(str(model_pkl_path))
        model_pkl_path.unlink(missing_ok=True)
        results_path = root / "metrics.txt"
        results_path.write_text(
            f"train_accuracy={train_accuracy:.4f}\ntest_accuracy={test_accuracy:.4f}\n"
            f"train_f1_weighted={train_f1:.4f}\ntest_f1_weighted={test_f1:.4f}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(results_path))
        results_path.unlink(missing_ok=True)

        print(
            f"[RandomForest] train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}"
        )
        print(f"train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")
        print("Run saved to MLflow.")


def run_dvc(X_train, X_test, y_train, y_test, model_dir: pathlib.Path, scaler, **kwargs):
    """Train and save to model_dir for DVC (no MLflow). Returns metrics dict."""
    model = RandomForestClassifier(
        max_depth=kwargs.get("max_depth", 10),
        n_estimators=kwargs.get("n_estimators", 100),
        random_state=kwargs.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_dir / "model.pkl")
    return {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "train_f1_weighted": float(f1_score(y_train, y_pred_train, average="weighted")),
        "test_f1_weighted": float(f1_score(y_test, y_pred_test, average="weighted")),
    }
