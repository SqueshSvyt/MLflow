"""
Мінімальні сирі CSV у форматі HAM10000 лише для юніт-тестів (load_combined, prepare).
Не використовується в GitHub Actions — CI тягне data/raw через DVC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DX_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
N_PER_CLASS = 12


def write_mini_ham_raw(out: Path) -> None:
    """Записує HAM10000_metadata.csv та hmnist_28_28_L.csv у каталог out."""
    out = out.resolve()
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

    pd.DataFrame(rows).to_csv(out / "HAM10000_metadata.csv", index=False)
    pd.DataFrame(hmnist_rows).to_csv(out / "hmnist_28_28_L.csv", index=False)
