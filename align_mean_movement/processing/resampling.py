from __future__ import annotations

import numpy as np

def resample_sequence_nan_robust(frames: np.ndarray, start: int, end: int, n_frames: int) -> np.ndarray:
    """
    Resample frames[start:end+1] to n_frames using linear interpolation per marker coordinate.
    frames: (T,108,3)
    returns: (n_frames,108,3)
    """
    seg = frames[start:end+1]
    Tseg = seg.shape[0]
    if Tseg == 1:
        return np.repeat(seg, n_frames, axis=0)

    t_old = np.linspace(0.0, 1.0, Tseg)
    t_new = np.linspace(0.0, 1.0, n_frames)
    out = np.full((n_frames, seg.shape[1], 3), np.nan, dtype=float)

    # For each marker and coord, interpolate over finite points
    for m in range(seg.shape[1]):
        for c in range(3):
            y = seg[:, m, c]
            finite = np.isfinite(y)
            if finite.sum() < 2:
                # keep NaN or constant if one point exists
                if finite.sum() == 1:
                    out[:, m, c] = y[finite][0]
                continue
            out[:, m, c] = np.interp(t_new, t_old[finite], y[finite])
    return out


# ----------------------------
# Plotly animation
# ----------------------------
