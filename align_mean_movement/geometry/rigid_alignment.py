from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from .rotations import rotation_matrix_from_axis_angle

def nanmean_keepdims(x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=axis, keepdims=keepdims)

def nanmedian(x: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.nanmedian(x))

def rms(x: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.nanmean(np.square(x))))

def trimmed_indices(residuals: np.ndarray, trim_frac: float) -> np.ndarray:
    """Return boolean mask of inliers as best (1-trim_frac) residuals. residuals shape (N,)."""
    n = residuals.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(round((1.0 - trim_frac) * n)))
    order = np.argsort(residuals)
    inl = np.zeros((n,), dtype=bool)
    inl[order[:k]] = True
    return inl

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform (R,t) s.t. R P + t ~= Q, using Kabsch.
    P,Q: Nx3 with finite values.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t

def apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t

def trimmed_ransac_rigid(P: np.ndarray, Q: np.ndarray, trials: int, subset: int, trim_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Robust rigid transform using trimmed-consensus RANSAC.
    Returns R,t,inlier_mask,trimmed_rms
    """
    n = P.shape[0]
    if n < max(3, subset):
        raise ValueError(f"Not enough correspondences for RANSAC: {n}")
    best_score = float("inf")
    best_R = None
    best_t = None
    best_inl = None

    idx_all = np.arange(n)
    for _ in range(trials):
        idx = rng.choice(idx_all, size=subset, replace=False)
        try:
            R, t = kabsch(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        P2 = apply_rt(P, R, t)
        res = np.linalg.norm(P2 - Q, axis=1)
        inl = trimmed_indices(res, trim_frac)
        score = rms(res[inl])
        if score < best_score:
            best_score = score
            best_R, best_t, best_inl = R, t, inl

    if best_R is None:
        raise ValueError("RANSAC failed to produce a valid transform.")

    # Refit on inliers
    R, t = kabsch(P[best_inl], Q[best_inl])
    P2 = apply_rt(P, R, t)
    res = np.linalg.norm(P2 - Q, axis=1)
    inl = trimmed_indices(res, trim_frac)
    score = rms(res[inl])
    return R, t, inl, score

def huber_irls_refine(P: np.ndarray, Q: np.ndarray, R0: np.ndarray, t0: np.ndarray, inl_mask: np.ndarray, iters: int, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple IRLS-like refinement: iteratively reweight points by Huber weights and solve weighted Kabsch
    on inliers only. Guard against NaNs and SVD non-convergence.
    """
    R, t = R0.copy(), t0.copy()
    P_inl = P[inl_mask]
    Q_inl = Q[inl_mask]
    if P_inl.shape[0] < 3:
        return R, t

    w = np.ones((P_inl.shape[0],), dtype=float)
    for _ in range(iters):
        P2 = apply_rt(P_inl, R, t)
        r = np.linalg.norm(P2 - Q_inl, axis=1)
        # Huber weights
        w = np.where(r <= k, 1.0, (k / (r + 1e-12)))
        if not np.isfinite(w).all():
            break
        # weighted centroids
        sw = np.sum(w)
        if sw < 1e-12:
            break
        Pc = np.sum(P_inl * w[:, None], axis=0) / sw
        Qc = np.sum(Q_inl * w[:, None], axis=0) / sw
        X = P_inl - Pc
        Y = Q_inl - Qc
        H = (X * w[:, None]).T @ Y
        if not np.isfinite(H).all():
            break
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            break
        Rn = Vt.T @ U.T
        if np.linalg.det(Rn) < 0:
            Vt[-1, :] *= -1
            Rn = Vt.T @ U.T
        tn = Qc - Rn @ Pc
        R, t = Rn, tn
    return R, t

def maybe_yaw_flip(P: np.ndarray, Q: np.ndarray, R: np.ndarray, t: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """Try a 180° flip about one axis in target space before applying R,t; keep best by RMS."""
    axis = axis.upper()
    if axis not in ("X","Y","Z"):
        return R, t, False, float("inf")

    Rflip = np.eye(3)
    ax = {"X": np.array([1,0,0]), "Y": np.array([0,1,0]), "Z": np.array([0,0,1])}[axis]
    Rflip = rotation_matrix_from_axis_angle(ax, math.pi)

    # Option A: no flip
    resA = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
    scoreA = rms(resA)

    # Option B: flip P first
    Pf = (Rflip @ P.T).T
    resB = np.linalg.norm(apply_rt(Pf, R, t) - Q, axis=1)
    scoreB = rms(resB)

    if scoreB < scoreA:
        # absorb flip into R: R*(Rflip*P) = (R*Rflip)P
        return (R @ Rflip), t, True, scoreB
    return R, t, False, scoreA


# ----------------------------
# Movement extraction + temporal resampling
# ----------------------------
