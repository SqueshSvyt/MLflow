"""Unit / smoke: валідація структури сирих та підготовлених даних."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data import load_combined
from tests.data_validation import (
    validate_hmnist_schema,
    validate_metadata_schema,
    validate_params_json,
    validate_prepared_split,
)
from tests.fixtures.mini_ham_testdata import write_mini_ham_raw


def test_validate_metadata_schema_ok():
    df = pd.DataFrame(
        {
            "lesion_id": ["a"],
            "image_id": ["i1"],
            "dx": ["nv"],
            "dx_type": ["x"],
            "age": [40.0],
            "sex": ["male"],
            "localization": ["arm"],
        }
    )
    validate_metadata_schema(df)


def test_validate_metadata_missing_column():
    df = pd.DataFrame({"dx": ["nv"]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_metadata_schema(df)


def test_load_combined_mini_fixture(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    raw = tmp_path / "raw"
    write_mini_ham_raw(raw)
    meta = raw / "HAM10000_metadata.csv"
    hmnist = raw / "hmnist_28_28_L.csv"
    mdf = pd.read_csv(meta)
    hdf = pd.read_csv(hmnist)
    validate_metadata_schema(mdf)
    validate_hmnist_schema(hdf, len(mdf))
    X, y, names = load_combined(meta, hmnist)
    assert X.shape[0] == len(mdf)
    assert len(names) == 787


def test_validate_prepared_after_prepare(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    raw = tmp_path / "raw"
    prep = tmp_path / "prepared"
    write_mini_ham_raw(raw)
    subprocess.run(
        [sys.executable, "src/prepare.py", str(raw), str(prep), "--test_size", "0.2"],
        cwd=root,
        check=True,
    )
    tr = pd.read_parquet(prep / "train.parquet")
    te = pd.read_parquet(prep / "test.parquet")
    validate_prepared_split(tr, te)
    validate_params_json(prep / "params.json")
    params = json.loads((prep / "params.json").read_text())
    assert params["test_size"] == 0.2
