"""
Станція Prepare для DVC-пайплайну.
Читає data/raw (метадані + hmnist), будує об'єднані дані, ділить на train/test, зберігає в data/prepared.
"""
import argparse
import json
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data import load_combined


def main():
    parser = argparse.ArgumentParser(description="Prepare: raw -> train/test splits")
    parser.add_argument("raw_dir", type=pathlib.Path, help="data/raw")
    parser.add_argument("prepared_dir", type=pathlib.Path, help="data/prepared")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    prepared_dir.mkdir(parents=True, exist_ok=True)

    meta_path = raw_dir / "HAM10000_metadata.csv"
    hmnist_path = raw_dir / "hmnist_28_28_L.csv"
    X, y, feature_names = load_combined(meta_path, hmnist_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["dx"] = y_train.values
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["dx"] = y_test.values

    train_df.to_parquet(prepared_dir / "train.parquet", index=False)
    test_df.to_parquet(prepared_dir / "test.parquet", index=False)
    split_params = {"test_size": args.test_size, "random_state": args.random_state}
    (prepared_dir / "params.json").write_text(json.dumps(split_params, indent=2), encoding="utf-8")
    print(f"Saved train {len(train_df)} and test {len(test_df)} to {prepared_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
