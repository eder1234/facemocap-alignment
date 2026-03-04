from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

def infer_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = [c.strip() for c in df.columns]
    lower = {c: c.lower() for c in cols}

    # path column
    path_candidates = ["complete_filepath", "filepath", "path", "csv_path", "relpath", "relative_path", "file", "fullpath"]
    path_col = None
    for cand in path_candidates:
        for c in cols:
            if lower[c] == cand:
                path_col = c
                break
        if path_col:
            break
    if path_col is None:
        raise ValueError("Could not infer file path column in metadata (expected complete_filepath or similar).")

    # movement column
    mov_candidates = ["facial_movement", "movement", "mov", "m", "single_movement", "movement_id"]
    mov_col = None
    for cand in mov_candidates:
        for c in cols:
            if lower[c] == cand:
                mov_col = c
                break
        if mov_col:
            break
    if mov_col is None:
        raise ValueError("Could not infer movement column in metadata (expected facial_movement or similar).")

    # group / condition column
    grp_candidates = ["condition", "group", "label", "is_healthy"]
    grp_col = None
    for cand in grp_candidates:
        for c in cols:
            if lower[c] == cand:
                grp_col = c
                break
        if grp_col:
            break
    if grp_col is None:
        raise ValueError("Could not infer condition/group column in metadata (expected condition).")

    return path_col, mov_col, grp_col

def normalize_movement(m) -> str:
    """
    Returns movement label like 'M1'..'M5' from either numeric or string.
    """
    if isinstance(m, str):
        s = m.strip()
        if s.upper().startswith("M"):
            return "M" + "".join([ch for ch in s[1:] if ch.isdigit()])
        # numeric string
        if s.isdigit():
            return f"M{int(s)}"
        return s
    try:
        return f"M{int(m)}"
    except Exception:
        return str(m)

def is_healthy_from_condition(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("healthy", "sain", "control", "controls", "0", "false?")  # conservative

# ----------------------------
# Main
# ----------------------------
