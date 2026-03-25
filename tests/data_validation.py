"""
Перевірки якості/структури даних (використовуються unit-тестами та CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_META_COLS = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
HAM_DX_VALUES = frozenset({"bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"})


def validate_metadata_schema(meta: pd.DataFrame) -> None:
    missing = REQUIRED_META_COLS - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    if meta["dx"].isna().any():
        raise ValueError("metadata dx contains NaN")
    unknown = set(meta["dx"].astype(str).unique()) - HAM_DX_VALUES
    if unknown:
        raise ValueError(f"unknown dx labels: {unknown}")


def validate_hmnist_schema(hmnist: pd.DataFrame, n_meta_rows: int) -> None:
    if len(hmnist) != n_meta_rows:
        raise ValueError(f"hmnist rows {len(hmnist)} != metadata rows {n_meta_rows}")
    pixel_cols = [c for c in hmnist.columns if c.startswith("pixel")]
    if len(pixel_cols) != 784:
        raise ValueError(f"expected 784 pixel columns, got {len(pixel_cols)}")


def validate_prepared_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if "dx" not in train.columns or "dx" not in test.columns:
        raise ValueError("prepared data must contain dx column")
    feat_train = [c for c in train.columns if c != "dx"]
    feat_test = [c for c in test.columns if c != "dx"]
    if feat_train != feat_test:
        raise ValueError("train and test feature columns differ")
    if len(feat_train) < 787:
        raise ValueError(f"expected >=787 feature cols (784+meta), got {len(feat_train)}")
    if train[feat_train].isna().all().any():
        raise ValueError("train has all-NaN feature column")
    if test[feat_test].isna().all().any():
        raise ValueError("test has all-NaN feature column")


def validate_params_json(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "test_size" not in data or "random_state" not in data:
        raise ValueError("params.json must contain test_size and random_state")
