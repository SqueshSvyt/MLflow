"""Shared utilities for all models: plotting, constants, MLflow reproducibility tags."""

import pathlib
import subprocess
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For PyTorch models (image + metadata)
IMG_FEATURES = 784
META_FEATURES = 3


def plot_feature_importance(
    model, feature_names, save_path: pathlib.Path, model_type: str, top_n: int = 50
):
    """Plot feature importance (top_n features when there are many)."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        coef = np.abs(model.coef_)
        imp = coef.mean(axis=0)
    order = np.argsort(imp)[::-1]
    n_show = min(top_n, len(imp))
    order = order[:n_show]
    fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.15)))
    ax.barh(range(n_show), imp[order], align="center")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=6)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Feature importance ({model_type}, top {n_show})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: pathlib.Path, model_type: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.unique(np.concatenate([y_true, y_pred])),
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion matrix ({model_type})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def log_reproducibility_mlflow(
    root: pathlib.Path,
    *,
    dataset_version: str = "v1",
    random_state: int | None = None,
    test_size: float | None = None,
    split_random_state: int | None = None,
) -> None:
    """Логує в поточний MLflow run теги/параметри для відтворюваності."""
    import mlflow

    mlflow.set_tag("dataset_version", dataset_version)
    if random_state is not None:
        mlflow.log_param("random_state", random_state)
    if test_size is not None:
        mlflow.log_param("test_size", test_size)
    if split_random_state is not None:
        mlflow.log_param("split_random_state", split_random_state)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            mlflow.set_tag("code_commit", out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
