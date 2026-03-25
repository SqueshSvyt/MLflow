#!/usr/bin/env python3
"""
Міні-датасет для CI: ті самі колонки, що HAM10000 + hmnist_28_28_L (784 пікселі + label).
Запуск з кореня репозиторію:
  python tests/fixtures/generate_mini_ham10000.py --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DX_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
N_PER_CLASS = 12  # достатньо для stratify у prepare


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    args = p.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []
    hmnist_rows = []
    pixel_names = [f"pixel{i:04d}" for i in range(784)]

    idx = 0
    for dx in DX_CLASSES:
        for _ in range(N_PER_CLASS):
            rows.append(
                {
                    "lesion_id": f"L{idx}",
                    "image_id": f"ISIC_{idx:06d}",
                    "dx": dx,
                    "dx_type": "UNK",
                    "age": float(30 + (idx % 50)),
                    "sex": ["male", "female"][idx % 2],
                    "localization": ["arm", "leg", "trunk"][idx % 3],
                }
            )
            pixels = rng.integers(0, 255, size=784, dtype=np.int64)
            row = {name: int(pixels[j]) for j, name in enumerate(pixel_names)}
            row["label"] = DX_CLASSES.index(dx)
            hmnist_rows.append(row)
            idx += 1

    meta = pd.DataFrame(rows)
    hmnist = pd.DataFrame(hmnist_rows)
    meta.to_csv(out / "HAM10000_metadata.csv", index=False)
    hmnist.to_csv(out / "hmnist_28_28_L.csv", index=False)
    print(f"Wrote {len(meta)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
