from __future__ import annotations

import numpy as np

def displacement_energy(frames_aligned: np.ndarray, neutral_idx: int, facial_slice: slice) -> np.ndarray:
    """
    Energy per frame based on facial displacement magnitude from neutral.
    Returns E shape (T,)
    """
    P0 = frames_aligned[neutral_idx, facial_slice, :]  # 105x3
    P = frames_aligned[:, facial_slice, :]
    with np.errstate(all="ignore"):
        d = P - P0[None, :, :]
        mag = np.sqrt(np.nansum(d**2, axis=-1))  # T,105
        E = np.nanmean(mag, axis=1)  # T
    return E

def extract_active_window(E: np.ndarray, neutral_idx: int, thr_percentile: float = 70.0, min_len: int = 10) -> Tuple[int,int]:
    """
    Extract contiguous active window around the maximum energy.
    Uses threshold as percentile of E (finite values). Returns [start,end] inclusive indices.
    """
    finite = np.isfinite(E)
    if finite.sum() == 0:
        return 0, len(E)-1
    thr = np.nanpercentile(E, thr_percentile)
    peak = int(np.nanargmax(E))
    active = (E >= thr) & finite
    if not active.any():
        return 0, len(E)-1

    # Find contiguous segment containing peak; else fallback to peak-neighborhood
    idx = np.where(active)[0]
    # Build segments
    segments = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((s, prev))
            s = i
            prev = i
    segments.append((s, prev))

    seg = None
    for a,b in segments:
        if a <= peak <= b:
            seg = (a,b)
            break
    if seg is None:
        seg = max(segments, key=lambda ab: ab[1]-ab[0])

    start, end = seg
    if (end - start + 1) < min_len:
        # Expand symmetrically
        half = (min_len - (end - start + 1)) // 2 + 1
        start = max(0, start - half)
        end = min(len(E)-1, end + half)
    return int(start), int(end)
