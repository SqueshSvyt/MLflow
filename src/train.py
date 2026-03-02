"""
Скрипт навчання моделей з логуванням експериментів у MLflow.
Датасет: HAM10000 — завжди зображення (hmnist 28x28) + метадані (вік, стать, локалізація).
Моделі: RandomForest, GradientBoosting (sklearn), CNN, ResNet (PyTorch).
"""
import argparse
import pathlib
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_data_path():
    root = pathlib.Path(__file__).resolve().parent.parent
    raw = root / "data" / "raw" / "HAM10000_metadata.csv"
    alt = root / "data" / "HAM10000_metadata.csv"
    if raw.exists():
        return raw
    if alt.exists():
        return alt
    raise FileNotFoundError(
        "Not found HAM10000_metadata.csv. Put file in data/raw/ or data/."
    )


def get_hmnist_path():
    """Path to hmnist 28x28 grayscale (pixels + label). Rows align with metadata by index."""
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / "data" / "raw" / "hmnist_28_28_L.csv"
    if not path.exists():
        raise FileNotFoundError("Not found data/raw/hmnist_28_28_L.csv")
    return path


def load_combined_data(metadata_path: pathlib.Path, hmnist_path: pathlib.Path):
    """Load images (hmnist 28x28) + metadata (age, sex_enc, loc_enc). Rows aligned by index."""
    meta = pd.read_csv(metadata_path)
    hmnist = pd.read_csv(hmnist_path)
    if len(meta) != len(hmnist):
        raise ValueError(
            f"Metadata rows {len(meta)} != hmnist rows {len(hmnist)}. Align datasets."
        )
    y = meta["dx"].copy()
    # Pixels
    X_img = hmnist.drop(columns=["label"], errors="ignore").astype(np.float64)
    pixel_names = list(X_img.columns)
    # Metadata: age, sex, localization
    df = meta.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")
    le_sex = LabelEncoder()
    le_loc = LabelEncoder()
    df["sex_enc"] = le_sex.fit_transform(df["sex"].astype(str))
    df["loc_enc"] = le_loc.fit_transform(df["localization"].astype(str))
    X_meta = df[["age", "sex_enc", "loc_enc"]].astype(np.float64).values
    feature_names = pixel_names + ["age", "sex_enc", "loc_enc"]
    X_combined = np.hstack([X_img.values, X_meta])
    return X_combined, y, feature_names


def plot_feature_importance(model, feature_names, save_path: pathlib.Path, model_type: str, top_n: int = 50):
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(np.concatenate([y_true, y_pred])))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion matrix ({model_type})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def get_model_and_params(args):
    """Returns (model, dict params for mlflow, need_scale). Only for sklearn models."""
    model_type = args.model
    if model_type == "RandomForest":
        model = RandomForestClassifier(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )
        params = {"max_depth": args.max_depth, "n_estimators": args.n_estimators}
        return model, params, False
    if model_type == "GradientBoosting":
        model = GradientBoostingClassifier(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
        params = {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
        }
        return model, params, False
    if model_type in ("CNN", "ResNet"):
        raise ValueError("CNN and ResNet are handled by PyTorch path; use run_pytorch_train.")
    raise ValueError(f"Unknown model: {model_type}")


# --- PyTorch: Simple CNN and ResNet (image + metadata) ---

IMG_FEATURES = 784
META_FEATURES = 3
CNN_EMBED = 128
RESNET_EMBED = 512


class SimpleCNN(nn.Module):
    """CNN for 28x28 grayscale + optional metadata (age, sex_enc, loc_enc) -> 7 classes."""
    def __init__(self, num_classes: int = 7, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        in_fc = CNN_EMBED + (META_FEATURES if use_meta else 0)
        self.classifier = nn.Linear(in_fc, num_classes)

    def forward(self, x_img, x_meta=None):
        x = self.features(x_img)
        x = x.view(x.size(0), -1)
        if self.use_meta and x_meta is not None:
            x = torch.cat([x, x_meta], dim=1)
        return self.classifier(x)


class ResNet18WithMeta(nn.Module):
    """ResNet18 backbone (1ch 28x28) + concat metadata -> num_classes."""
    def __init__(self, num_classes: int = 7, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        backbone = tv_models.resnet18(weights=None, num_classes=RESNET_EMBED)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        in_fc = RESNET_EMBED + (META_FEATURES if use_meta else 0)
        self.fc = nn.Linear(in_fc, num_classes)

    def forward(self, x_img, x_meta=None):
        x = self.backbone(x_img)
        if self.use_meta and x_meta is not None:
            x = torch.cat([x, x_meta], dim=1)
        return self.fc(x)


def run_pytorch_train(args, root, X_train, X_test, y_train, y_test, le_dx):
    """Train CNN or ResNet on image + metadata, log to MLflow."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for CNN/ResNet. Install: pip install torch torchvision")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(le_dx.classes_)
    # X_train, X_test: (N, 787) = 784 pixels + 3 meta
    X_img_train = X_train[:, :IMG_FEATURES]
    X_meta_train = X_train[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES]
    X_img_test = X_test[:, :IMG_FEATURES]
    X_meta_test = X_test[:, IMG_FEATURES:IMG_FEATURES + META_FEATURES]

    def to_img_tensor(X):
        t = torch.from_numpy(X.astype(np.float32))
        return t.reshape(-1, 1, 28, 28)
    X_tr_img = to_img_tensor(X_img_train)
    X_tr_meta = torch.from_numpy(X_meta_train.astype(np.float32))
    y_tr = torch.from_numpy(le_dx.transform(y_train)).long()
    train_ds = TensorDataset(X_tr_img, X_tr_meta, y_tr)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.model == "CNN":
        model = SimpleCNN(num_classes=n_classes, use_meta=True).to(device)
    else:
        model = ResNet18WithMeta(num_classes=n_classes, use_meta=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("ham10000_baseline")
    with mlflow.start_run():
        mlflow.set_tag("model_type", args.model)
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("data_source", "images+metadata")
        mlflow.log_param("model", args.model)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)

        for epoch in range(args.epochs):
            model.train()
            for x_img, x_meta, yb in train_loader:
                x_img = x_img.to(device)
                x_meta = x_meta.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_img, x_meta), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        X_te_img = to_img_tensor(X_img_test).to(device)
        X_te_meta = torch.from_numpy(X_meta_test.astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(X_te_img, X_te_meta)
            y_pred_idx = logits.argmax(dim=1).cpu().numpy()
        y_pred_test = le_dx.inverse_transform(y_pred_idx)
        X_tr_img_eval = to_img_tensor(X_img_train).to(device)
        X_tr_meta_eval = torch.from_numpy(X_meta_train.astype(np.float32)).to(device)
        with torch.no_grad():
            y_pred_train_idx = model(X_tr_img_eval, X_tr_meta_eval).argmax(dim=1).cpu().numpy()
        y_pred_train = le_dx.inverse_transform(y_pred_train_idx)

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

        cm_path = root / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, cm_path, args.model)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        import mlflow.pytorch as mlflow_pt
        import pickle
        mlflow_pt.log_model(model, "model", registered_model_name=None)
        with open(root / "label_encoder.pkl", "wb") as f:
            pickle.dump(le_dx, f)
        mlflow.log_artifact(str(root / "label_encoder.pkl"), "model")
        (root / "label_encoder.pkl").unlink(missing_ok=True)
        results_path = root / "metrics.txt"
        results_path.write_text(
            f"train_accuracy={train_accuracy:.4f}\ntest_accuracy={test_accuracy:.4f}\n"
            f"train_f1_weighted={train_f1:.4f}\ntest_f1_weighted={test_f1:.4f}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(results_path))
        results_path.unlink(missing_ok=True)

        print(f"[{args.model}] train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}")
        print(f"train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")
        print("Run saved to MLflow.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train model with MLflow (CLI: model type + hyperparameters)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "GradientBoosting", "CNN", "ResNet"],
        help="Тип моделі",
    )
    parser.add_argument("--max_depth", type=int, default=5, help="max_depth (RF, GB)")
    parser.add_argument("--n_estimators", type=int, default=100, help="n_estimators (RF, GB)")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning_rate (GB, CNN, ResNet)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for CNN/ResNet")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for CNN/ResNet")
    parser.add_argument("--test_size", type=float, default=0.2, help="Частка тестової вибірки")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--author", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="v1")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent

    # Завжди зображення + метадані (вік, стать, локалізація)
    meta_path = get_data_path()
    hmnist_path = get_hmnist_path()
    X, y, feature_names = load_combined_data(meta_path, hmnist_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if args.model in ("CNN", "ResNet"):
        le_dx = LabelEncoder()
        le_dx.fit(y)
        run_pytorch_train(args, root, X_train, X_test, y_train, y_test, le_dx)
        return 0

    model, model_params, _ = get_model_and_params(args)

    mlflow.set_experiment("ham10000_baseline")

    with mlflow.start_run():
        mlflow.set_tag("model_type", args.model)
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("data_source", "images+metadata")

        mlflow.log_param("model", args.model)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        for k, v in model_params.items():
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
        plot_feature_importance(model, feature_names, importance_path, args.model, top_n=25)
        mlflow.log_artifact(str(importance_path))
        importance_path.unlink(missing_ok=True)

        cm_path = root / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, cm_path, args.model)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        # Save only model (for LR scaling can be repeated at inference)
        mlflow.sklearn.log_model(model, "model")

        results_path = root / "metrics.txt"
        results_path.write_text(
            f"train_accuracy={train_accuracy:.4f}\ntest_accuracy={test_accuracy:.4f}\n"
            f"train_f1_weighted={train_f1:.4f}\ntest_f1_weighted={test_f1:.4f}\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(results_path))
        results_path.unlink(missing_ok=True)

        print(f"[{args.model}] train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}")
        print(f"train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")
        print("Run saved to MLflow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
