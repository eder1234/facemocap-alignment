from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from .rigid_alignment import rms

def is_finite(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=-1)

@dataclass
class DentalFrame:
    R: np.ndarray  # 3x3 rotation matrix
    o: np.ndarray  # 3 translation origin

def compute_dental_frame(d: np.ndarray) -> Optional[DentalFrame]:
    """
    Compute a right-handed orthonormal frame from 3 dental markers d shape (3,3).
    Returns None if degenerate / NaN.
    """
    if d.shape != (3,3):
        raise ValueError("dental markers must be (3,3)")
    if not np.isfinite(d).all():
        return None
    o = d.mean(axis=0)

    x = d[1] - d[0]
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return None
    x = x / nx

    v = d[2] - d[0]
    # normal to dental plane
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return None
    z = z / nz

    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        return None
    y = y / ny

    R = np.stack([x, y, z], axis=1)  # columns are basis vectors in world coords
    # Ensure R is proper rotation
    if np.linalg.det(R) < 0:
        # Flip z to make det positive
        R[:, 2] *= -1.0
    return DentalFrame(R=R, o=o)

def world_to_dental(frames_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert each frame to dental coordinates using per-frame dental frame.
    Returns:
      frames_dental: (T,108,3) in dental coords
      ok_mask: (T,) whether dental frame was valid
    """
    T = frames_world.shape[0]
    out = np.full_like(frames_world, np.nan, dtype=float)
    ok = np.zeros((T,), dtype=bool)
    for t in range(T):
        d = frames_world[t, 0:3, :]
        fr = compute_dental_frame(d)
        if fr is None:
            continue
        ok[t] = True
        # P_d = R^T (P - o)
        out[t] = (fr.R.T @ (frames_world[t] - fr.o).T).T
    return out, ok

def dental_marker_drift(frames_dental: np.ndarray) -> float:
    """RMS drift of the 3 dental markers in dental coordinates over time (should be small)."""
    d = frames_dental[:, 0:3, :]  # T,3,3
    # center per frame
    with np.errstate(all="ignore"):
        d0 = d - np.nanmean(d, axis=1, keepdims=True)
    # compute per-frame magnitude
    mag = np.sqrt(np.nansum(d0**2, axis=-1))  # T,3
    return rms(mag)


# ----------------------------
# Neutral selection (new)
# ----------------------------
