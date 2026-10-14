"""Compare regenerated analysis outputs with the committed versions.

CSV files are compared numerically with a relative tolerance, because the last
digits of bootstrap CIs and very small p-values differ between NumPy/SciPy
builds (for example macOS vs Linux). Non-numeric columns and the LaTeX tables in
paper/tables/ (rounded to three decimals) must match exactly.

Usage (from the repository root, after regenerating):
    python scripts/compare_outputs.py
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-9


def committed(path: Path) -> str:
    return subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def compare_csv(path: Path) -> list[str]:
    old = pd.read_csv(io.StringIO(committed(path)))
    new = pd.read_csv(ROOT / path)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return [f"{path}: shape or columns differ"]
    problems = []
    for column in old.columns:
        if pd.api.types.is_numeric_dtype(old[column]) and pd.api.types.is_numeric_dtype(new[column]):
            if not np.allclose(old[column], new[column], rtol=RTOL, atol=0.0, equal_nan=True):
                worst = np.nanmax(np.abs(old[column] - new[column]) / np.maximum(np.abs(old[column]), 1e-300))
                problems.append(f"{path}: column {column} differs (max relative difference {worst:.2e})")
        elif not old[column].astype(str).equals(new[column].astype(str)):
            problems.append(f"{path}: column {column} differs")
    return problems


def main() -> int:
    problems = []
    for path in sorted((ROOT / "results/analysis").glob("*.csv")):
        problems += compare_csv(path.relative_to(ROOT))
    for path in sorted((ROOT / "paper/tables").glob("*.tex")):
        relative = path.relative_to(ROOT)
        if path.read_text() != committed(relative):
            problems.append(f"{relative}: differs")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"All analysis outputs match the committed versions (relative tolerance {RTOL:g}); paper tables match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
