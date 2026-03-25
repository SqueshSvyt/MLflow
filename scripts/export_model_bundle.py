#!/usr/bin/env python3
"""
Експорт артефактів після підготовки даних:
  - model.pkl (модель + scaler у joblib dict)
  - metrics.json (accuracy / f1)
  - confusion_matrix.png

Читає data/prepared (train.parquet, test.parquet).
Вихідний каталог: env CI_ARTIFACT_DIR або за замовчуванням ci_artifacts/.

Реалізація: делегує до src/ci_export_rf.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ci_export = REPO / "src" / "ci_export_rf.py"
    if not ci_export.exists():
        print(f"Missing {ci_export}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(ci_export), run_name="__main__")


if __name__ == "__main__":
    main()
