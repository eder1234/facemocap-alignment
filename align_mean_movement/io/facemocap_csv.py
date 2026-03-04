from __future__ import annotations

import numpy as np
import pandas as pd

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_facemocap_csv(csv_path: Path, skiprows: int = 5, drop_first_cols: int = 2) -> np.ndarray:
    """
    Returns: frames x 108 x 3 float array with NaNs where missing.
    Assumes after dropping first `drop_first_cols`, remaining columns are 108*3.
    """
    df = pd.read_csv(csv_path, skiprows=skiprows, header=None)
    if df.shape[1] <= drop_first_cols:
        raise ValueError(f"CSV has too few columns: {df.shape[1]} in {csv_path}")
    arr = df.iloc[:, drop_first_cols:].to_numpy(dtype=float, copy=True)
    ncols = arr.shape[1]
    if ncols % 3 != 0:
        raise ValueError(f"Expected columns multiple of 3 after dropping {drop_first_cols}, got {ncols} in {csv_path}")
    n_markers = ncols // 3
    if n_markers < 108:
        # Allow fewer markers but expect at least 108 in this project; warn by error to avoid silent misparse.
        raise ValueError(f"Expected at least 108 markers (324 cols), got {n_markers} markers in {csv_path}")
    if n_markers > 108:
        # Some exports may include extra columns; keep first 108 markers.
        arr = arr[:, :108*3]
        n_markers = 108
    frames = arr.reshape(arr.shape[0], n_markers, 3)
    return frames


# ----------------------------
# Dental frame conversion
# ----------------------------
