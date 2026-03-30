from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_hardness_column(columns: List[str]) -> Optional[str]:
    """Heuristic: pick the column whose name contains Mohs/hardness."""
    lowered = {c: c.lower().strip() for c in columns}
    candidates = []
    for col, low in lowered.items():
        if "hardness" in low or "mohs" in low:
            candidates.append(col)
    if not candidates:
        return None
    # Prefer columns that explicitly mention hardness.
    hardness_priority = sorted(candidates, key=lambda c: ("hardness" not in lowered[c], len(c)))
    return hardness_priority[0]


def _load_csv(path: Path) -> pd.DataFrame:
    # Mendeley CSVs are typically standard comma-separated.
    return pd.read_csv(path)


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    # Keep numeric columns other than the target.
    # Dataset also includes some non-numeric columns (e.g., crystal system/structure).
    # Also drop "Unnamed: 0" index-like columns that come from CSV exports.
    numeric_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.lower().startswith("unnamed"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # Sometimes numeric columns come in as strings; keep only if coercion yields many numbers.
            coerced = _to_numeric_series(df[c])
            if coerced.notna().mean() >= 0.9:
                numeric_cols.append(c)

    # The paper states 11 compositional descriptors; if we have more, try to trim by dropping
    # any obviously-non-descriptor columns (by name). Otherwise keep all numeric.
    if len(numeric_cols) > 11:
        blacklist = {"class", "crystal", "system", "structure", "label", "name"}
        trimmed = []
        for c in numeric_cols:
            low = c.lower()
            if any(b in low for b in blacklist):
                continue
            trimmed.append(c)
        if len(trimmed) == 11:
            return trimmed
        if len(trimmed) > 11:
            # Fallback: take first 11 stable-by-name.
            return sorted(trimmed)[:11]
    return sorted(numeric_cols)


@dataclass(frozen=True)
class LoadedSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    hardness_column: str


def load_mendeley_train_test(data_dir: str | Path, train_rows: int, test_rows: int) -> LoadedSplit:
    """
    Load the Mendeley dataset by auto-detecting which CSV is train (622 rows) vs test (51 rows).

    Assumes:
    - Each CSV contains a hardness column (name contains "hardness" or "mohs")
    - Numeric feature columns are the 11 compositional descriptors (heuristically selected).
    """
    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.glob("**/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    # First pass: read each CSV and detect hardness column + row count.
    candidates: List[Tuple[Path, int, str]] = []
    parsed: dict[Path, Tuple[pd.DataFrame, str]] = {}
    for p in csv_paths:
        df = _load_csv(p)
        target_col = _find_hardness_column(list(df.columns))
        if target_col is None:
            continue
        parsed[p] = (df, target_col)
        candidates.append((p, len(df), target_col))

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not identify both train and test CSVs. "
            "Please ensure the extracted dataset contains two CSVs with a hardness column."
        )

    # Match by row count (622/51). If that fails, fall back to largest as train.
    train_path = None
    test_path = None
    for p, n, _ in candidates:
        if n == train_rows:
            train_path = p
        if n == test_rows:
            test_path = p

    if train_path is None or test_path is None:
        # Fallback: choose train as the largest candidate, test as the smallest.
        by_size = sorted(candidates, key=lambda t: t[1])
        test_path = by_size[0][0]
        train_path = by_size[-1][0]

    train_df, train_target_col = parsed[train_path]
    test_df, test_target_col = parsed[test_path]

    feature_cols = _select_feature_columns(train_df, target_col=train_target_col)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns detected.")

    # Build matrices; coerce numeric.
    X_train = train_df[feature_cols].apply(_to_numeric_series).to_numpy(dtype=float)
    y_train = _to_numeric_series(train_df[train_target_col]).to_numpy(dtype=float)

    X_test = test_df[feature_cols].apply(_to_numeric_series).to_numpy(dtype=float)
    y_test = _to_numeric_series(test_df[test_target_col]).to_numpy(dtype=float)

    return LoadedSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        hardness_column=train_target_col,
    )

