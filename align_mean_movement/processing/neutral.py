from __future__ import annotations

import math
import numpy as np

def pick_neutral_frame_most_complete(frames_dental: np.ndarray, first_pct: float, facial_slice: slice) -> int:
    """
    Pick the frame with maximum number of finite facial markers among first `first_pct` of frames.
    """
    T = frames_dental.shape[0]
    n_first = max(1, int(math.ceil(first_pct * T)))
    sub = frames_dental[:n_first, facial_slice, :]  # n_first,105,3
    finite = np.isfinite(sub).all(axis=-1)  # n_first,105
    counts = finite.sum(axis=1)  # n_first
    best = int(np.argmax(counts))
    if counts[best] == 0:
        raise ValueError("Neutral selection failed: no finite facial markers in early frames after dental-frame conversion.")
    return best


# ----------------------------
# Robust rigid alignment (facial only)
# ----------------------------
