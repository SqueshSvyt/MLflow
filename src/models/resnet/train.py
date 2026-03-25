"""ResNet18 model (image + metadata): train and log to MLflow or run_dvc for DVC."""

import pathlib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import mlflow

from ..common import plot_confusion_matrix, IMG_FEATURES, META_FEATURES, log_reproducibility_mlflow

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models as tv_models

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    tv_models = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

RESNET_EMBED = 512

if TORCH_AVAILABLE:

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


def run(args, root, X_train, X_test, y_train, y_test, le_dx):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for ResNet. Install: pip install torch torchvision")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(le_dx.classes_)

    X_img_train = X_train[:, :IMG_FEATURES]
    X_meta_train = X_train[:, IMG_FEATURES : IMG_FEATURES + META_FEATURES]
    X_img_test = X_test[:, :IMG_FEATURES]
    X_meta_test = X_test[:, IMG_FEATURES : IMG_FEATURES + META_FEATURES]

    def to_img_tensor(X):
        t = torch.from_numpy(X.astype(np.float32))
        return t.reshape(-1, 1, 28, 28)

    X_tr_img = to_img_tensor(X_img_train)
    X_tr_meta = torch.from_numpy(X_meta_train.astype(np.float32))
    y_tr = torch.from_numpy(le_dx.transform(y_train)).long()
    train_ds = TensorDataset(X_tr_img, X_tr_meta, y_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ResNet18WithMeta(num_classes=n_classes, use_meta=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("ham10000_baseline")
    with mlflow.start_run():
        log_reproducibility_mlflow(
            root,
            dataset_version=args.dataset_version,
            random_state=args.random_state,
            test_size=getattr(args, "test_size", None),
            split_random_state=getattr(args, "split_random_state", None),
        )
        mlflow.set_tag("model_type", "ResNet")
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("data_source", "images+metadata")
        mlflow.log_param("model", "ResNet")
        if getattr(args, "test_size", None) is not None:
            mlflow.log_param("test_size", args.test_size)
        if getattr(args, "split_random_state", None) is not None:
            mlflow.log_param("split_random_state", args.split_random_state)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)

        for epoch in range(args.epochs):
            model.train()
            for x_img, x_meta, yb in train_loader:
                x_img, x_meta, yb = x_img.to(device), x_meta.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_img, x_meta), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        X_te_img = to_img_tensor(X_img_test).to(device)
        X_te_meta = torch.from_numpy(X_meta_test.astype(np.float32)).to(device)
        with torch.no_grad():
            y_pred_idx = model(X_te_img, X_te_meta).argmax(dim=1).cpu().numpy()
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
        plot_confusion_matrix(y_test, y_pred_test, cm_path, "ResNet")
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        import mlflow.pytorch as mlflow_pt

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

        print(f"[ResNet] train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}")
        print(f"train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")
        print("Run saved to MLflow.")


def run_dvc(X_train, X_test, y_train, y_test, model_dir: pathlib.Path, le_dx, **kwargs):
    """Train and save to model_dir for DVC (no MLflow). Returns metrics dict."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for ResNet. pip install torch torchvision")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(le_dx.classes_)
    epochs = kwargs.get("epochs", 10)
    batch_size = kwargs.get("batch_size", 64)
    lr = kwargs.get("learning_rate", 1e-3)

    X_img_train = X_train[:, :IMG_FEATURES]
    X_meta_train = X_train[:, IMG_FEATURES : IMG_FEATURES + META_FEATURES]
    X_img_test = X_test[:, :IMG_FEATURES]
    X_meta_test = X_test[:, IMG_FEATURES : IMG_FEATURES + META_FEATURES]

    def to_img_tensor(X):
        t = torch.from_numpy(X.astype(np.float32))
        return t.reshape(-1, 1, 28, 28)

    X_tr_img = to_img_tensor(X_img_train)
    X_tr_meta = torch.from_numpy(X_meta_train.astype(np.float32))
    y_tr = torch.from_numpy(le_dx.transform(y_train)).long()
    train_ds = TensorDataset(X_tr_img, X_tr_meta, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ResNet18WithMeta(num_classes=n_classes, use_meta=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x_img, x_meta, yb in train_loader:
            x_img, x_meta, yb = x_img.to(device), x_meta.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_img, x_meta), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    X_te_img = to_img_tensor(X_img_test).to(device)
    X_te_meta = torch.from_numpy(X_meta_test.astype(np.float32)).to(device)
    with torch.no_grad():
        y_pred_idx = model(X_te_img, X_te_meta).argmax(dim=1).cpu().numpy()
    y_pred_test = le_dx.inverse_transform(y_pred_idx)
    X_tr_img_eval = to_img_tensor(X_img_train).to(device)
    X_tr_meta_eval = torch.from_numpy(X_meta_train.astype(np.float32)).to(device)
    with torch.no_grad():
        y_pred_train_idx = model(X_tr_img_eval, X_tr_meta_eval).argmax(dim=1).cpu().numpy()
    y_pred_train = le_dx.inverse_transform(y_pred_train_idx)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le_dx, f)
    return {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "train_f1_weighted": float(f1_score(y_train, y_pred_train, average="weighted")),
        "test_f1_weighted": float(f1_score(y_test, y_pred_test, average="weighted")),
    }
