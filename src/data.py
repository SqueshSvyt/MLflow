"""
Спільне завантаження та попередня обробка даних HAM10000.
Використовується в prepare.py і train.py — без дублювання логіки.
"""
import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_combined(metadata_path: pathlib.Path, hmnist_path: pathlib.Path):
    """
    Завантажити об'єднані дані: зображення (hmnist 28x28) + метадані (age, sex_enc, loc_enc).
    Рядки збігаються за індексом.
    Повертає: X (np.ndarray), y (Series dx), feature_names (list).
    """
    meta = pd.read_csv(metadata_path)
    hmnist = pd.read_csv(hmnist_path)
    if len(meta) != len(hmnist):
        raise ValueError(
            f"Metadata rows {len(meta)} != hmnist rows {len(hmnist)}. Align datasets."
        )

    y = meta["dx"].copy()
    X_img = hmnist.drop(columns=["label"], errors="ignore").astype(np.float64)
    feature_names = list(X_img.columns) + ["age", "sex_enc", "loc_enc"]

    df = meta.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")
    le_sex = LabelEncoder()
    le_loc = LabelEncoder()
    df["sex_enc"] = le_sex.fit_transform(df["sex"].astype(str))
    df["loc_enc"] = le_loc.fit_transform(df["localization"].astype(str))
    X_meta = df[["age", "sex_enc", "loc_enc"]].astype(np.float64).values

    X = np.hstack([X_img.values, X_meta])
    return X, y, feature_names


def get_raw_data_paths(root: pathlib.Path):
    """Повертає (metadata_path, hmnist_path) для data/raw відносно root."""
    raw = root / "data" / "raw"
    meta = raw / "HAM10000_metadata.csv"
    alt_meta = root / "data" / "HAM10000_metadata.csv"
    metadata_path = meta if meta.exists() else alt_meta
    hmnist_path = raw / "hmnist_28_28_L.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("Not found HAM10000_metadata.csv in data/raw or data/.")
    if not hmnist_path.exists():
        raise FileNotFoundError("Not found data/raw/hmnist_28_28_L.csv")
    return metadata_path, hmnist_path
