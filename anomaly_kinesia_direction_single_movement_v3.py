#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly quantification for a single FaceMoCap movement sample.

Pipeline (high level):
1) Lookup the target sample in the metadata CSV via its complete_filepath -> get movement Mk and labels.
2) Load target CSV -> extract movement window and resample to n_frames.
3) Load reference mean_healthy.npy for that Mk (already n_frames).
4) Express both sequences in a per-frame dental coordinate system (head-motion removed).
5) Drop dental markers, align reference->target in neutral pose using robust rigid alignment
   (fixed 90° rotations search + optional yaw flip + Huber + trimming), then propagate to all frames.
6) Optional time alignment (none | lag | dtw).
7) Compute movement-focused metrics:
   A) RMSE of delta displacements (per-marker, global)
   B) Amplitude (max displacement) deficit/excess (per-marker, global)
   C) Time-to-peak shift (per-marker, global)
8) Write reports + plots + synchronized Plotly HTML visualization with top-K worst marker connections.

Assumptions (configurable):
- CSV numeric data begins after 5 header rows (default) and markers are stored as contiguous XYZ columns.
- The first `dental_n` markers are dental support markers (excluded from facial alignment and scoring).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib (no seaborn)
import matplotlib.pyplot as plt

# Prefer canonical movement normalization from the project when available
try:
    from align_mean_movement.io.metadata import normalize_movement as _normalize_movement  # type: ignore
except Exception:
    _normalize_movement = None


# Plotly for interactive HTML
try:
    import plotly.graph_objects as go
except Exception:
    go = None


def normalize_movement(value: object) -> str:
    """Normalize movement label from metadata to folder name like 'M5'.

    - If align_mean_movement.io.metadata.normalize_movement is available, use it.
    - Otherwise, handle common cases locally: 'M5', 5, 5.0, '5.0', 'M5bis' -> 'M5'.
    """
    if _normalize_movement is not None:
        try:
            m = _normalize_movement(value)
            # Ensure canonical form 'M<digits>'
            s = str(m).strip()
            mm = re.search(r"(M\d+)", s.upper())
            if mm:
                return mm.group(1)
        except Exception:
            pass

    if value is None:
        raise ValueError("Movement is None")
    s = str(value).strip()
    mm = re.search(r"(M\d+)", s.upper())
    if mm:
        return mm.group(1)
    try:
        f = float(s)
        mi = int(round(f))
        if abs(f - mi) < 1e-6:
            return f"M{mi}"
    except Exception:
        pass
    mm2 = re.search(r"(\d+)", s)
    if mm2:
        return f"M{int(mm2.group(1))}"
    raise ValueError(f"Could not normalize movement value: {value!r}")


# -----------------------------
# IO helpers
# -----------------------------

def safe_id_from_filepath(fp: str) -> str:
    h = hashlib.sha1(fp.encode("utf-8")).hexdigest()[:12]
    base = re.sub(r"[^a-zA-Z0-9]+", "_", Path(fp).stem)[:40].strip("_")
    if not base:
        base = "sample"
    return f"{base}_{h}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_metadata_row(metadata_csv: str, complete_filepath: str) -> pd.Series:
    df = pd.read_csv(metadata_csv)
    if "complete_filepath" not in df.columns:
        raise ValueError("metadata CSV must contain a 'complete_filepath' column.")
    m = df["complete_filepath"].astype(str) == str(complete_filepath)
    if not m.any():
        # Try normalizing slashes
        norm = str(complete_filepath).replace("\\", "/")
        m = df["complete_filepath"].astype(str).str.replace("\\\\", "/", regex=True) == norm
    if not m.any():
        # Try substring match (dangerous) - only if unique
        mm = df["complete_filepath"].astype(str).str.contains(re.escape(str(complete_filepath)), regex=True)
        if mm.sum() == 1:
            return df[mm].iloc[0]
        raise FileNotFoundError("complete_filepath not found in metadata.")
    if m.sum() > 1:
        # If duplicates exist, take first and warn in summary later.
        return df[m].iloc[0]
    return df[m].iloc[0]

def load_facemocap_csv_points(
    csv_path: str,
    skiprows: int = 5,
    usecols_start: int = 2,
    usecols_end: Optional[int] = 326,
) -> np.ndarray:
    """
    Returns array X with shape (T, N, 3), dtype float64, with NaNs preserved.
    Default column slicing matches your earlier usage: usecols=range(2, 326) -> 324 cols -> 108 points.
    """
    csv_path = str(csv_path)
    if usecols_end is None:
        df = pd.read_csv(csv_path, skiprows=skiprows, header=None)
        df = df.iloc[:, usecols_start:]
    else:
        df = pd.read_csv(csv_path, skiprows=skiprows, header=None, usecols=list(range(usecols_start, usecols_end)))
    arr = df.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] % 3 != 0:
        raise ValueError(f"Unexpected CSV numeric shape {arr.shape}. Expected (T, 3*N).")
    n_pts = arr.shape[1] // 3
    X = arr.reshape(arr.shape[0], n_pts, 3)
    return X

def load_mean_healthy_npy(path: str) -> np.ndarray:
    X = np.load(path)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"mean_healthy.npy must be (T,N,3), got {X.shape}")
    return X.astype(np.float64)

def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

# -----------------------------
# Geometry / alignment
# -----------------------------

def valid_mask_points(P: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,) bool, True if finite in all coords."""
    return np.isfinite(P).all(axis=-1)

def dental_frame_transform(dental_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dental frame (R, o) from 3 dental points.
    - dental_pts: (3,3) in world coords
    Returns:
      R: (3,3) with columns as axes in world coords
      o: (3,) origin (centroid)
    """
    if dental_pts.shape != (3, 3):
        raise ValueError("dental_frame_transform expects exactly 3 dental points.")
    o = np.nanmean(dental_pts, axis=0)
    p0, p1, p2 = dental_pts[0], dental_pts[1], dental_pts[2]
    x = p1 - p0
    nx = np.linalg.norm(x)
    if not np.isfinite(nx) or nx < 1e-8:
        raise ValueError("Invalid dental markers: cannot define x axis.")
    x = x / nx
    v = p2 - p0
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if not np.isfinite(nz) or nz < 1e-8:
        raise ValueError("Invalid dental markers: cannot define z axis.")
    z = z / nz
    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if not np.isfinite(ny) or ny < 1e-8:
        raise ValueError("Invalid dental markers: cannot define y axis.")
    y = y / ny
    R = np.stack([x, y, z], axis=1)  # columns
    return R, o

def to_dental_coords(Xw: np.ndarray, dental_n: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Express every frame in its own dental frame:
      Xd[t] = R(t)^T * (Xw[t] - o(t))
    Returns:
      Xd: (T,N,3)
      R_all: (T,3,3)
      o_all: (T,3)
    If dental transform cannot be computed for a frame (NaNs), we set that frame's Xd to NaN.
    """
    T, N, _ = Xw.shape
    if N < dental_n:
        raise ValueError("Not enough points for dental markers.")
    Xd = np.full_like(Xw, np.nan, dtype=np.float64)
    R_all = np.full((T, 3, 3), np.nan, dtype=np.float64)
    o_all = np.full((T, 3), np.nan, dtype=np.float64)
    for t in range(T):
        dent = Xw[t, :dental_n, :]
        if not np.isfinite(dent).all():
            continue
        try:
            R, o = dental_frame_transform(dent)
        except Exception:
            continue
        # Transform: x_d = R^T (x - o)
        Xd[t] = (Xw[t] - o) @ R  # because (x-o) is row vec; multiply by R (cols axes) gives coords in that basis
        R_all[t] = R
        o_all[t] = o
    return Xd, R_all, o_all

def rot90_matrix(ax: str, k: int) -> np.ndarray:
    """Rotation by k*90 degrees about axis 'X','Y','Z'."""
    k = k % 4
    if k == 0:
        return np.eye(3)
    ang = k * (math.pi / 2.0)
    c, s = math.cos(ang), math.sin(ang)
    if ax.upper() == "X":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
    if ax.upper() == "Y":
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
    if ax.upper() == "Z":
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    raise ValueError("axis must be X,Y,Z")

def yaw_flip_matrix(axis: str) -> np.ndarray:
    axis = axis.upper()
    if axis == "X":
        return np.diag([-1, 1, 1]).astype(np.float64)
    if axis == "Y":
        return np.diag([1, -1, 1]).astype(np.float64)
    if axis == "Z":
        return np.diag([1, 1, -1]).astype(np.float64)
    raise ValueError("yaw_flip_axis must be X/Y/Z")

def weighted_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted Kabsch: find R,t minimizing sum w_i || R P_i + t - Q_i ||^2
    P,Q: (M,3), w: (M,)
    Returns R (3,3), t (3,)
    """
    w = w.astype(np.float64)
    w = np.clip(w, 0.0, None)
    sw = w.sum()
    if sw <= 0:
        raise ValueError("All weights are zero in weighted_kabsch.")
    muP = (P * w[:, None]).sum(axis=0) / sw
    muQ = (Q * w[:, None]).sum(axis=0) / sw
    X = P - muP
    Y = Q - muQ
    H = (X * w[:, None]).T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    return R, t

def robust_rigid_fit(
    P: np.ndarray,
    Q: np.ndarray,
    trim_frac: float = 0.10,
    huber_iters: int = 3,
    huber_k: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Robust rigid fit using iterative reweighting:
    - Start with w=1
    - Iteratively:
      * Kabsch
      * residuals r
      * trim top trim_frac residuals -> w=0 for those
      * Huber reweighting for remaining
    Returns: R,t, final weights w, weighted_rmse
    """
    M = P.shape[0]
    w = np.ones(M, dtype=np.float64)

    def robust_scale(r: np.ndarray) -> float:
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med))
        s = 1.4826 * mad
        if not np.isfinite(s) or s < 1e-9:
            s = np.nanstd(r)
        if not np.isfinite(s) or s < 1e-9:
            s = 1.0
        return float(s)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    for _ in range(max(1, huber_iters)):
        R, t = weighted_kabsch(P, Q, w)
        res = np.linalg.norm((P @ R.T) + t - Q, axis=1)  # careful: P rowvec, so apply R as (R p) -> p @ R^T
        # Trim
        if 0.0 < trim_frac < 0.49:
            ktrim = int(math.floor((1.0 - trim_frac) * M))
            if ktrim < 3:
                ktrim = 3
            keep_idx = np.argsort(res)[:ktrim]
            w_trim = np.zeros_like(w)
            w_trim[keep_idx] = 1.0
        else:
            w_trim = np.ones_like(w)
        # Huber
        s = robust_scale(res[w_trim > 0])
        c = huber_k * s
        w_h = np.ones_like(w)
        if np.isfinite(c) and c > 0:
            big = res > c
            w_h[big] = (c / (res[big] + 1e-12))
        w = w_trim * w_h

    # Final score
    R, t = weighted_kabsch(P, Q, w)
    res = np.linalg.norm((P @ R.T) + t - Q, axis=1)
    sw = w.sum()
    wrmse = float(math.sqrt((w * (res ** 2)).sum() / max(sw, 1e-12)))
    return R, t, w, wrmse

def best_neutral_face_alignment(
    Pref: np.ndarray,
    Qtar: np.ndarray,
    trim_frac: float,
    huber_iters: int,
    huber_k: float,
    fixed_rot_step_xyz: Tuple[int, int, int] = (90, 90, 90),
    try_yaw_flip: bool = True,
    yaw_flip_axis: str = "Z",
) -> Dict[str, object]:
    """
    Search over fixed 90-degree rotations (grid defined by step) and optional yaw flip,
    then robustly fit rigid transform.
    Returns dict with best R,t, candidate meta, score.
    """
    sx, sy, sz = fixed_rot_step_xyz
    def angles(step: int) -> List[int]:
        if step <= 0:
            return [0]
        vals = list(range(0, 360, step))
        if 0 not in vals:
            vals = [0] + vals
        return sorted(set([v % 360 for v in vals]))

    ax_list = angles(sx)
    ay_list = angles(sy)
    az_list = angles(sz)

    flip_mats = [np.eye(3)]
    if try_yaw_flip:
        flip_mats.append(yaw_flip_matrix(yaw_flip_axis))

    best = None
    for Fx in flip_mats:
        for ax in ax_list:
            Rx = rot90_matrix("X", ax // 90) if ax % 90 == 0 else rot90_matrix("X", 0)
            for ay in ay_list:
                Ry = rot90_matrix("Y", ay // 90) if ay % 90 == 0 else rot90_matrix("Y", 0)
                for az in az_list:
                    Rz = rot90_matrix("Z", az // 90) if az % 90 == 0 else rot90_matrix("Z", 0)
                    R0 = Fx @ (Rz @ (Ry @ Rx))
                    P0 = Pref @ R0.T  # apply to row vectors
                    try:
                        Rfit, tfit, w, score = robust_rigid_fit(P0, Qtar, trim_frac=trim_frac, huber_iters=huber_iters, huber_k=huber_k)
                    except Exception:
                        continue
                    # Overall transform: x -> Rfit*(R0*x) + tfit = (Rfit*R0)*x + tfit
                    Rtot = Rfit @ R0
                    cand = dict(
                        R=Rtot,
                        t=tfit,
                        score=score,
                        flip=(not np.allclose(Fx, np.eye(3))),
                        yaw_flip_axis=yaw_flip_axis if (not np.allclose(Fx, np.eye(3))) else None,
                        fixed_rot_xyz=(ax, ay, az),
                        inlier_weight_sum=float(w.sum()),
                    )
                    if best is None or score < best["score"]:
                        best = cand
    if best is None:
        raise RuntimeError("Failed to find a valid neutral facial alignment.")
    return best

# -----------------------------
# Movement windowing / resampling
# -----------------------------

def motion_energy_from_neutral(X: np.ndarray, neutral_idx: int, face_idx: np.ndarray) -> np.ndarray:
    """Energy curve per frame: median norm displacement from neutral over facial markers."""
    neutral = X[neutral_idx]
    dif = X - neutral[None, :, :]
    dif = dif[:, face_idx, :]
    # norms (T, M)
    n = np.linalg.norm(dif, axis=-1)
    # median over markers ignoring NaN
    e = np.nanmedian(n, axis=1)
    return e

def choose_neutral_idx(X: np.ndarray, face_idx: np.ndarray, neutral_first_pct: float = 0.05) -> int:
    T = X.shape[0]
    k = max(1, int(round(T * max(0.0, min(1.0, neutral_first_pct)))))
    k = min(T, max(k, 1))
    # Use first k frames; choose frame with minimal motion relative to frame0 (proxy)
    base = X[0]
    dif = X[:k] - base[None, :, :]
    n = np.linalg.norm(dif[:, face_idx, :], axis=-1)
    e = np.nanmedian(n, axis=1)
    idx = int(np.nanargmin(e))
    return idx

def extract_movement_window(
    X: np.ndarray,
    face_idx: np.ndarray,
    neutral_idx: int,
    energy_thr_percentile: float = 40.0,
    min_window_len: int = 30,
    max_gap: int = 5,
) -> Tuple[int, int, np.ndarray]:
    """
    Find a contiguous movement window where energy exceeds a percentile threshold.
    Allows small gaps (<= max_gap) inside active window.
    Returns (start,end,energy_curve) with end exclusive.
    """
    e = motion_energy_from_neutral(X, neutral_idx, face_idx)
    thr = np.nanpercentile(e, energy_thr_percentile)
    active = e >= thr
    # Fill small gaps
    if max_gap > 0:
        active_f = active.copy()
        gap = 0
        for t in range(len(active_f)):
            if active_f[t]:
                gap = 0
            else:
                gap += 1
                if gap <= max_gap:
                    active_f[t] = True
        active = active_f
    idx = np.where(active)[0]
    if idx.size == 0:
        # fallback: take last min_window_len frames if possible
        s = max(0, X.shape[0] - min_window_len)
        return s, X.shape[0], e
    s = int(idx[0])
    eidx = int(idx[-1]) + 1
    # Ensure minimum length
    if (eidx - s) < min_window_len:
        mid = (s + eidx) // 2
        half = min_window_len // 2
        s = max(0, mid - half)
        eidx = min(X.shape[0], s + min_window_len)
        s = max(0, eidx - min_window_len)
    return s, eidx, e

def resample_linear(X: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear resampling along time axis with NaN-aware interpolation per coordinate."""
    T, N, C = X.shape
    if T == n_frames:
        return X.copy()
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, n_frames)
    Y = np.full((n_frames, N, C), np.nan, dtype=np.float64)
    for i in range(N):
        for c in range(C):
            y = X[:, i, c]
            m = np.isfinite(y)
            if m.sum() < 2:
                continue
            Y[:, i, c] = np.interp(t_new, t_old[m], y[m])
    return Y

# -----------------------------
# Time alignment
# -----------------------------

def delta_sequence(X: np.ndarray, t0: int, face_idx: np.ndarray) -> np.ndarray:
    """(T,N,3) -> (T,M,3) facial deltas relative to t0."""
    Xf = X[:, face_idx, :]
    X0 = X[t0, face_idx, :]
    return Xf - X0[None, :, :]

def global_delta_descriptor(D: np.ndarray) -> np.ndarray:
    """(T,M,3) -> (T,) 1D descriptor: median displacement magnitude over markers."""
    mag = np.linalg.norm(D, axis=-1)
    return np.nanmedian(mag, axis=1)

def apply_lag_alignment(ref: np.ndarray, tar: np.ndarray, lag_max: int) -> Tuple[np.ndarray, int, float]:
    """
    ref, tar: (T,M,3) deltas
    Shift ref by lag to best match tar (integer lag). Positive lag means ref is advanced (ref[t+lag] vs tar[t]).
    Returns warped_ref (T,M,3), best_lag, best_score
    """
    T = tar.shape[0]
    best_lag = 0
    best_score = float("inf")

    def score_for_lag(lag: int) -> float:
        # overlap region
        if lag >= 0:
            t0, t1 = 0, T - lag
            A = ref[lag:lag + (t1 - t0)]
            B = tar[t0:t1]
        else:
            lag2 = -lag
            t0, t1 = lag2, T
            A = ref[0:(t1 - t0)]
            B = tar[t0:t1]
        dif = A - B
        mag2 = np.nansum(dif * dif, axis=-1)  # (Tov,M)
        # global mean over markers/time
        val = np.nanmean(mag2)
        if not np.isfinite(val):
            return float("inf")
        return float(math.sqrt(val))

    for lag in range(-lag_max, lag_max + 1):
        sc = score_for_lag(lag)
        if sc < best_score:
            best_score = sc
            best_lag = lag

    # Build warped ref to length T by shifting with NaN padding
    warped = np.full_like(ref, np.nan)
    if best_lag >= 0:
        warped[0:T - best_lag] = ref[best_lag:T]
    else:
        lag2 = -best_lag
        warped[lag2:T] = ref[0:T - lag2]
    return warped, best_lag, best_score

def dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[List[Tuple[int,int]], float]:
    """Classic DTW path for 1D arrays a,b. Returns path and cost."""
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    # backtrack
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = np.argmin([D[i-1,j], D[i,j-1], D[i-1,j-1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path, float(D[n,m])

def apply_dtw_alignment(ref: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    DTW-align ref deltas to tar deltas using 1D descriptor (median displacement magnitude).
    Returns warped_ref of shape (T,M,3) and info dict (path, cost).
    """
    d_ref = global_delta_descriptor(ref)
    d_tar = global_delta_descriptor(tar)
    path, cost = dtw_path(d_ref, d_tar)
    T, M, _ = tar.shape
    # For each target time j, average all ref frames i mapped to it
    buckets: Dict[int, List[int]] = {j: [] for j in range(T)}
    for i, j in path:
        if 0 <= j < T:
            buckets[j].append(i)
    warped = np.full_like(tar, np.nan)
    for j in range(T):
        idx = buckets.get(j, [])
        if len(idx) == 0:
            continue
        warped[j] = np.nanmean(ref[idx], axis=0)
    info = {"dtw_cost": cost, "dtw_path_len": len(path)}
    return warped, info

# -----------------------------
# Metrics A/B/C
# -----------------------------

@dataclass
class MetricResults:
    per_marker: pd.DataFrame
    per_frame: pd.DataFrame
    summary: Dict[str, object]

def compute_metrics_ABC(
    Xref: np.ndarray,  # (T,N,3) aligned in dental coords
    Xtar: np.ndarray,  # (T,N,3) target in dental coords
    face_idx: np.ndarray,
    neutral_ref: int,
    neutral_tar: int,
    time_align: str = "lag",
    lag_max: int = 10,
    amp_ref_min_mode: str = "percentile",
    amp_ref_min_percentile: float = 20.0,
    amp_ref_min: float = 0.5,
) -> MetricResults:
    """
    Compute:
      A: RMSE of delta displacements per marker + global
      B: amplitude max displacement per marker (ref/tar, ratio, diff)
      C: time-to-peak (argmax amplitude) per marker (ref/tar, diff)
    Includes optional time alignment (none|lag|dtw) on delta trajectories.
    """
    Dref = delta_sequence(Xref, neutral_ref, face_idx)  # (T,M,3)
    Dtar = delta_sequence(Xtar, neutral_tar, face_idx)  # (T,M,3)

    time_info = {"time_align": time_align}
    if time_align == "none":
        Dref_w = Dref
        align_score = None
    elif time_align == "lag":
        Dref_w, best_lag, best_score = apply_lag_alignment(Dref, Dtar, lag_max=lag_max)
        time_info.update({"best_lag": int(best_lag), "lag_score_rmse": float(best_score)})
        align_score = best_score
    elif time_align == "dtw":
        Dref_w, dtwinfo = apply_dtw_alignment(Dref, Dtar)
        time_info.update(dtwinfo)
        align_score = dtwinfo.get("dtw_cost", None)
    else:
        raise ValueError("time_align must be one of: none, lag, dtw")

    # A) RMSE delta per marker
    dif = Dtar - Dref_w
    e = np.linalg.norm(dif, axis=-1)  # (T,M)
    rmse_i = np.sqrt(np.nanmean(e**2, axis=0))
    rmse_global = float(np.sqrt(np.nanmean(e**2)))

    # Per-frame global RMSE
    rmse_frame = np.sqrt(np.nanmean(e**2, axis=1))

    # B) amplitude
    amp_ref = np.nanmax(np.linalg.norm(Dref_w, axis=-1), axis=0)
    amp_tar = np.nanmax(np.linalg.norm(Dtar, axis=-1), axis=0)
    amp_ratio = amp_tar / (amp_ref + 1e-9)
    amp_diff = amp_tar - amp_ref

    # Kinesia scores (movement-focused)
    # Gate to avoid unstable ratios when the healthy reference amplitude is near-zero.
    amp_ref_valid = amp_ref[np.isfinite(amp_ref) & (amp_ref > 0)]
    if amp_ref_valid.size == 0:
        amp_ref_gate = 0.0
    else:
        if amp_ref_min_mode == "percentile":
            amp_ref_gate = float(np.nanpercentile(amp_ref_valid, amp_ref_min_percentile))
        else:
            amp_ref_gate = float(amp_ref_min)
    amp_ref_gate = max(amp_ref_gate, 1e-9)

    # signed log-ratio: <0 hypokinesia (under-movement), >0 hyperkinesia (over-movement)
    log_amp_ratio = np.log((amp_tar + 1e-9) / (amp_ref + 1e-9))
    two_sided_amp_dev = np.abs(log_amp_ratio)

    # validity for kinesia-based ranking
    kinesia_ok = np.isfinite(log_amp_ratio) & np.isfinite(amp_ref) & (amp_ref >= amp_ref_gate)


    # D) Directional disagreement ("counter-direction" markers)
    # Compute cosine similarity between delta vectors; negative values indicate opposite hemispheres.
    mag_ref = np.linalg.norm(Dref_w, axis=-1)  # (T,M)
    mag_tar = np.linalg.norm(Dtar, axis=-1)    # (T,M)
    dot = np.nansum(Dref_w * Dtar, axis=-1)    # (T,M)
    denom = (mag_ref * mag_tar) + 1e-9
    cos_sim = dot / denom  # (T,M)

    # Gate: only consider frames where both ref and target are moving meaningfully.
    # Use the same reference amplitude gate, and mirror it for target via the same numeric value.
    dir_ok = np.isfinite(cos_sim) & (mag_ref >= amp_ref_gate) & (mag_tar >= amp_ref_gate)

    opp_tau = 0.2  # cosine threshold for "opposite" direction (angle > ~101 degrees)
    opp_mask = dir_ok & (cos_sim < -opp_tau)

    # Per-marker: fraction of opposite frames among valid frames
    dir_valid_counts = np.sum(dir_ok, axis=0)
    opp_counts = np.sum(opp_mask, axis=0)
    opp_fraction = np.where(dir_valid_counts > 0, opp_counts / np.maximum(dir_valid_counts, 1), np.nan)

    # Weighted opposite score: average of max(0, -cos_sim) weighted by min(mag_ref, mag_tar)
    w = np.minimum(mag_ref, mag_tar)
    w = np.where(dir_ok, w, np.nan)
    neg_part = np.where(dir_ok, np.maximum(0.0, -cos_sim), np.nan)
    opp_score = np.nansum(w * neg_part, axis=0) / (np.nansum(w, axis=0) + 1e-9)

    cos_median = np.nanmedian(np.where(dir_ok, cos_sim, np.nan), axis=0)

    # C) time-to-peak
    # (if all NaN, return NaN)
    mag_ref = np.linalg.norm(Dref_w, axis=-1)  # (T,M)
    mag_tar = np.linalg.norm(Dtar, axis=-1)
    tpeak_ref = np.full(mag_ref.shape[1], np.nan)
    tpeak_tar = np.full(mag_ref.shape[1], np.nan)
    for i in range(mag_ref.shape[1]):
        if np.isfinite(mag_ref[:, i]).any():
            tpeak_ref[i] = int(np.nanargmax(mag_ref[:, i]))
        if np.isfinite(mag_tar[:, i]).any():
            tpeak_tar[i] = int(np.nanargmax(mag_tar[:, i]))
    tpeak_diff = tpeak_tar - tpeak_ref

    marker_ids = face_idx.astype(int).tolist()  # original indices in N
    per_marker = pd.DataFrame({
        "marker_id": marker_ids,
        "rmse_delta": rmse_i,
        "amp_ref": amp_ref,
        "amp_tar": amp_tar,
        "amp_ratio": amp_ratio,
        "log_amp_ratio": log_amp_ratio,
        "two_sided_amp_dev": two_sided_amp_dev,
        "kinesia_ok": kinesia_ok,
        "dir_ok_frames": dir_valid_counts,
        "opp_fraction": opp_fraction,
        "opp_score": opp_score,
        "cos_median": cos_median,
        "amp_diff": amp_diff,
        "tpeak_ref": tpeak_ref,
        "tpeak_tar": tpeak_tar,
        "tpeak_diff": tpeak_diff,
        "valid_frames": np.sum(np.isfinite(e), axis=0),
    })

    per_frame = pd.DataFrame({
        "frame_idx": np.arange(Dtar.shape[0], dtype=int),
        "rmse_delta_frame": rmse_frame,
        "valid_markers": np.sum(np.isfinite(e), axis=1),
    })

    # Headline aggregates (robust-ish: median)
    summary = {
        "rmse_global_delta": rmse_global,
        "rmse_median_marker": float(np.nanmedian(rmse_i)),
        "amp_ratio_median": float(np.nanmedian(amp_ratio)),
        "amp_diff_median": float(np.nanmedian(amp_diff)),
        "amp_ref_gate": float(amp_ref_gate),
        "kinesia_ok_count": int(np.sum(kinesia_ok)),
        "opp_score_median": float(np.nanmedian(opp_score)),
        "opp_fraction_median": float(np.nanmedian(opp_fraction)),
        "tpeak_diff_median": float(np.nanmedian(tpeak_diff)),
        "tpeak_diff_mean": float(np.nanmean(tpeak_diff)),
    }
    summary.update(time_info)
    return MetricResults(per_marker=per_marker, per_frame=per_frame, summary=summary)

# -----------------------------
# Plotting / visualization
# -----------------------------

def plot_rmse_per_marker(per_marker: pd.DataFrame, out_png: Path, top_k: int = 10) -> None:
    df = per_marker.sort_values("rmse_delta", ascending=False).reset_index(drop=True)
    y = df["rmse_delta"].to_numpy()
    x = np.arange(len(df))
    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Markers (sorted)")
    plt.ylabel("RMSE delta (mm or units)")
    plt.title("Per-marker delta RMSE (movement anomaly)")
    if top_k > 0:
        plt.axvline(top_k - 0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_rmse_over_time(per_frame: pd.DataFrame, out_png: Path) -> None:
    x = per_frame["frame_idx"].to_numpy()
    y = per_frame["rmse_delta_frame"].to_numpy()
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Frame")
    plt.ylabel("Global RMSE delta")
    plt.title("Global delta RMSE over time")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def make_plotly_overlay(
    Xtar: np.ndarray,  # (T,N,3) dental coords
    Xref: np.ndarray,  # (T,N,3) aligned dental coords
    face_idx: np.ndarray,
    topk_marker_ids: List[int],
    out_html: Path,
    title: str = "Target (red) vs Reference (blue) — synchronized",
) -> None:
    if go is None:
        return
    T, N, _ = Xtar.shape
    # We'll show only facial markers for clarity
    idx = face_idx
    # build frames
    frames = []
    for t in range(T):
        P = Xtar[t, idx, :]
        Q = Xref[t, idx, :]
        # marker ids in facial list, but topk_marker_ids are original marker ids (in N)
        # Map original marker ids -> index within idx
        id_to_local = {int(mid): j for j, mid in enumerate(idx.tolist())}
        line_x, line_y, line_z = [], [], []
        for mid in topk_marker_ids:
            if mid not in id_to_local:
                continue
            j = id_to_local[mid]
            p = P[j]
            q = Q[j]
            if not (np.isfinite(p).all() and np.isfinite(q).all()):
                continue
            line_x += [p[0], q[0], None]
            line_y += [p[1], q[1], None]
            line_z += [p[2], q[2], None]

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=Q[:,0], y=Q[:,1], z=Q[:,2], mode="markers",
                             marker=dict(size=3), name="Reference (aligned)", showlegend=(t==0)),
                go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], mode="markers",
                             marker=dict(size=3), name="Target", showlegend=(t==0)),
                go.Scatter3d(x=line_x, y=line_y, z=line_z, mode="lines",
                             line=dict(width=3), name=f"Top-{len(topk_marker_ids)} connections", showlegend=(t==0)),
            ],
            name=str(t)
        ))

    # initial data
    P0 = Xtar[0, idx, :]
    Q0 = Xref[0, idx, :]
    id_to_local = {int(mid): j for j, mid in enumerate(idx.tolist())}
    line_x, line_y, line_z = [], [], []
    for mid in topk_marker_ids:
        if mid not in id_to_local:
            continue
        j = id_to_local[mid]
        p, q = P0[j], Q0[j]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            continue
        line_x += [p[0], q[0], None]
        line_y += [p[1], q[1], None]
        line_z += [p[2], q[2], None]

    fig = go.Figure(
        data=[
            go.Scatter3d(x=Q0[:,0], y=Q0[:,1], z=Q0[:,2], mode="markers",
                         marker=dict(size=3), name="Reference (aligned)"),
            go.Scatter3d(x=P0[:,0], y=P0[:,1], z=P0[:,2], mode="markers",
                         marker=dict(size=3), name="Target"),
            go.Scatter3d(x=line_x, y=line_y, z=line_z, mode="lines",
                         line=dict(width=3), name=f"Top-{len(topk_marker_ids)} connections"),
        ],
        frames=frames
    )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]},
            ]
        }],
        sliders=[{
            "steps": [{"method": "animate", "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                       "label": str(k)} for k in range(T)],
            "currentvalue": {"prefix": "Frame: "}
        }]
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to facemocap_metadata.csv")
    ap.add_argument("--target_complete_filepath", required=True, help="Exact complete_filepath value to select target sample")
    ap.add_argument("--root_override", default=None, help="Optional root override to rewrite metadata filepath (prefix substitution)")
    ap.add_argument("--ref_root", required=True, help="Root directory containing outputs/Mk/mean_healthy.npy")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    # CSV parsing
    ap.add_argument("--skiprows", type=int, default=5)
    ap.add_argument("--usecols_start", type=int, default=2)
    ap.add_argument("--usecols_end", type=int, default=326, help="End column (exclusive). Use 0 to mean 'auto to end'.")
    ap.add_argument("--dental_n", type=int, default=3, help="Number of dental markers at beginning of marker list")

    # Windowing / resampling (target)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--neutral_first_pct", type=float, default=0.05)
    ap.add_argument("--energy_thr_percentile", type=float, default=40.0)
    ap.add_argument("--min_window_len", type=int, default=30)
    ap.add_argument("--max_gap", type=int, default=5)

    # Alignment options (neutral facial alignment)
    ap.add_argument("--trim_frac", type=float, default=0.10)
    ap.add_argument("--huber_iters", type=int, default=3)
    ap.add_argument("--huber_k", type=float, default=1.5)
    ap.add_argument("--try_yaw_flip", action="store_true")
    ap.add_argument("--yaw_flip_axis", default="Z", choices=["X","Y","Z"])
    ap.add_argument("--fixed_rot_xyz", nargs=3, type=int, default=[90,90,90], help="Rotation grid step in degrees (must be divisor of 360)")

    # Time alignment
    ap.add_argument("--time_align", default="lag", choices=["none","lag","dtw"])
    ap.add_argument("--lag_max", type=int, default=10)

    # Viz
    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--topk_mode", default="twosided", choices=["rmse","hypo","hyper","twosided"],
                    help="Which metric to use for selecting top-K markers in visualization lines. "
                         "rmse=delta RMSE, hypo=hypokinesia (lowest amp_ratio), hyper=hyperkinesia (highest amp_ratio), "
                         "twosided=two-sided amplitude deviation |log(amp_ratio)| (default).")
    ap.add_argument("--amp_ref_min_mode", default="percentile", choices=["percentile","absolute"],
                    help="How to set the minimum reference amplitude gate to avoid unstable ratios.")
    ap.add_argument("--amp_ref_min_percentile", type=float, default=20.0,
                    help="Percentile of amp_ref used as gate when amp_ref_min_mode=percentile (default 20).")
    ap.add_argument("--amp_ref_min", type=float, default=0.5,
                    help="Absolute minimum amp_ref (same units as data) when amp_ref_min_mode=absolute.")
    args = ap.parse_args()

    usecols_end = None if int(args.usecols_end) == 0 else int(args.usecols_end)

    # Metadata lookup
    row = read_metadata_row(args.metadata, args.target_complete_filepath)
    movement = None
    for col in ["facial_movement", "movement", "Mk"]:
        if col in row.index and pd.notna(row[col]):
            movement = str(row[col]).strip()
            break
    if movement is None:
        raise ValueError("Could not infer movement from metadata row. Expected column like 'facial_movement'.")
    movement = normalize_movement(movement)

    # Resolve target filepath possibly with root_override
    target_fp = str(args.target_complete_filepath)
    if args.root_override:
        # If metadata contains paths rooted elsewhere, allow simple prefix replacement:
        # replace the drive root up to "Data_FaceMoCap" with root_override if present.
        # fallback: if file does not exist, try join root_override with relative tail after Data_FaceMoCap
        if not os.path.exists(target_fp):
            marker = "Data_FaceMoCap"
            if marker in target_fp:
                rel = target_fp.split(marker, 1)[1].lstrip("/\\")
                cand = os.path.join(args.root_override, rel)
                if os.path.exists(cand):
                    target_fp = cand

    # Prepare output dir
    out_root = Path(args.out_dir)
    sid = safe_id_from_filepath(str(args.target_complete_filepath))
    out = out_root / movement / sid
    ensure_dir(out)

    # Identity
    identity = {
        "complete_filepath": str(args.target_complete_filepath),
        "resolved_csv_path": str(target_fp),
        "movement": movement,
        "topk_mode": str(args.topk_mode),
        "metadata_row": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.to_dict().items()},
        "cli_args": vars(args),
    }
    write_json(out / "target_identity.json", identity)

    # Load target and reference
    Xtar_w = load_facemocap_csv_points(target_fp, skiprows=args.skiprows, usecols_start=args.usecols_start, usecols_end=usecols_end)
    ref_path = Path(args.ref_root) / movement / "mean_healthy.npy"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference mean_healthy.npy not found at: {ref_path}")
    Xref_w = load_mean_healthy_npy(str(ref_path))

    # Ensure reference frames length
    if Xref_w.shape[0] != args.n_frames:
        Xref_w = resample_linear(Xref_w, args.n_frames)

    # Determine number of points and indices
    if Xtar_w.shape[1] != Xref_w.shape[1]:
        raise ValueError(f"Target N={Xtar_w.shape[1]} != Reference N={Xref_w.shape[1]}. Check CSV columns and mean_healthy generation.")
    N = Xtar_w.shape[1]
    if args.dental_n >= N:
        raise ValueError("dental_n must be < total number of markers.")
    face_idx = np.arange(args.dental_n, N, dtype=int)

    # Extract target movement window and resample
    neutral_idx_tar_raw = choose_neutral_idx(Xtar_w, face_idx, neutral_first_pct=args.neutral_first_pct)
    s, e, energy = extract_movement_window(
        Xtar_w, face_idx, neutral_idx_tar_raw,
        energy_thr_percentile=args.energy_thr_percentile,
        min_window_len=args.min_window_len,
        max_gap=args.max_gap
    )
    Xtar_win = Xtar_w[s:e]
    Xtar = resample_linear(Xtar_win, args.n_frames)

    # For reference, assume already movement window with neutral at frame0
    neutral_idx_ref = 0
    neutral_idx_tar = choose_neutral_idx(Xtar, face_idx, neutral_first_pct=args.neutral_first_pct)

    # Express in dental frame per frame (head motion removed)
    Xtar_d, Rtar_all, otar_all = to_dental_coords(Xtar, dental_n=args.dental_n)
    Xref_d, Rref_all, oref_all = to_dental_coords(Xref_w, dental_n=args.dental_n)

    # Neutral facial alignment in dental coordinates (exclude dental markers)
    Pref0 = Xref_d[neutral_idx_ref, face_idx, :]
    Qtar0 = Xtar_d[neutral_idx_tar, face_idx, :]

    m0 = valid_mask_points(Pref0) & valid_mask_points(Qtar0)
    if m0.sum() < 3:
        raise RuntimeError(f"Not enough valid facial markers in neutral for alignment (valid={int(m0.sum())}).")

    Pref0v = Pref0[m0]
    Qtar0v = Qtar0[m0]

    best = best_neutral_face_alignment(
        Pref0v, Qtar0v,
        trim_frac=args.trim_frac,
        huber_iters=args.huber_iters,
        huber_k=args.huber_k,
        fixed_rot_step_xyz=(int(args.fixed_rot_xyz[0]), int(args.fixed_rot_xyz[1]), int(args.fixed_rot_xyz[2])),
        try_yaw_flip=bool(args.try_yaw_flip),
        yaw_flip_axis=args.yaw_flip_axis,
    )
    R = best["R"]
    t = best["t"]
    # Propagate to all frames of ref in dental coordinates
    Xref_aligned = np.full_like(Xref_d, np.nan)
    for tt in range(Xref_d.shape[0]):
        P = Xref_d[tt]
        if not np.isfinite(P).any():
            continue
        # apply to each point row: R p + t -> p @ R^T + t
        Xref_aligned[tt] = (P @ R.T) + t[None, :]

    # Metrics (A/B/C) with time alignment
    metrics = compute_metrics_ABC(
        Xref=Xref_aligned,
        Xtar=Xtar_d,
        face_idx=face_idx,
        neutral_ref=neutral_idx_ref,
        neutral_tar=neutral_idx_tar,
        time_align=args.time_align,
        lag_max=args.lag_max,
        amp_ref_min_mode=args.amp_ref_min_mode,
        amp_ref_min_percentile=args.amp_ref_min_percentile,
        amp_ref_min=args.amp_ref_min,
    )

    # Identify top-K markers for visualization lines (selectable metric)
    topk = int(max(0, args.top_k))
    pm = metrics.per_marker.copy()

    if args.topk_mode == "rmse":
        pm_sorted = pm.sort_values("rmse_delta", ascending=False)
        topk_marker_ids = pm_sorted["marker_id"].head(topk).astype(int).tolist()
    elif args.topk_mode == "hypo":
        # hypokinesia: lowest amp_ratio among reliable markers
        pm2 = pm[pm["kinesia_ok"] == True].copy()
        pm_sorted = pm2.sort_values("amp_ratio", ascending=True)
        topk_marker_ids = pm_sorted["marker_id"].head(topk).astype(int).tolist()
    elif args.topk_mode == "hyper":
        # hyperkinesia: highest amp_ratio among reliable markers
        pm2 = pm[pm["kinesia_ok"] == True].copy()
        pm_sorted = pm2.sort_values("amp_ratio", ascending=False)
        topk_marker_ids = pm_sorted["marker_id"].head(topk).astype(int).tolist()
    else:
        # two-sided amplitude deviation: |log(amp_ratio)| among reliable markers
        pm2 = pm[pm["kinesia_ok"] == True].copy()
        pm_sorted = pm2.sort_values("two_sided_amp_dev", ascending=False)
        topk_marker_ids = pm_sorted["marker_id"].head(topk).astype(int).tolist()

    # Also generate ranked lists for analysis (regardless of visualization choice)
    pm_rmse = pm.sort_values("rmse_delta", ascending=False)
    pm_hypo = pm[pm["kinesia_ok"] == True].sort_values("amp_ratio", ascending=True)
    pm_hyper = pm[pm["kinesia_ok"] == True].sort_values("amp_ratio", ascending=False)
    pm_twosided = pm[pm["kinesia_ok"] == True].sort_values("two_sided_amp_dev", ascending=False)

    # Marker id lists for different ranking modes
    topk_ids_rmse = pm_rmse["marker_id"].head(topk).astype(int).tolist()
    topk_ids_hypo = pm_hypo["marker_id"].head(topk).astype(int).tolist()
    topk_ids_hyper = pm_hyper["marker_id"].head(topk).astype(int).tolist()
    topk_ids_twosided = pm_twosided["marker_id"].head(topk).astype(int).tolist()

    pm_counterdir = pm.sort_values("opp_score", ascending=False)
    topk_ids_counterdir = pm_counterdir["marker_id"].head(topk).astype(int).tolist()

    # Write reports
    write_csv(out / "per_marker_metrics.csv", metrics.per_marker)
    
    # Additional ranked lists for interpretability (palsy vs compensation)
    write_csv(out / "topk_rmse_delta.csv", pm_rmse.head(topk))
    write_csv(out / "topk_hypokinesia.csv", pm_hypo.head(topk))
    write_csv(out / "topk_hyperkinesia.csv", pm_hyper.head(topk))
    write_csv(out / "topk_two_sided_amp_dev.csv", pm_twosided.head(topk))
    write_csv(out / "topk_counter_direction.csv", pm_counterdir.head(topk))
    write_csv(out / "per_frame_metrics.csv", metrics.per_frame)

    summary = {
        "movement": movement,
        "topk_mode": str(args.topk_mode),
        "ref_path": str(ref_path),
        "amp_ref_gate": float(metrics.summary.get("amp_ref_gate", float("nan"))),
        "kinesia_ok_count": int(metrics.summary.get("kinesia_ok_count", 0)),
        "target_complete_filepath": str(args.target_complete_filepath),
        "target_resolved_csv_path": str(target_fp),
        "target_window_raw": {"start": int(s), "end": int(e), "neutral_idx_raw": int(neutral_idx_tar_raw)},
        "neutral_idx_tar_resampled": int(neutral_idx_tar),
        "neutral_idx_ref": int(neutral_idx_ref),
        "alignment": {
            "R": np.asarray(R).tolist(),
            "t": np.asarray(t).tolist(),
            "score": float(best["score"]),
            "flip": bool(best["flip"]),
            "yaw_flip_axis": best.get("yaw_flip_axis", None),
            "fixed_rot_xyz": best.get("fixed_rot_xyz", None),
            "inlier_weight_sum": float(best.get("inlier_weight_sum", 0.0)),
            "dental_n_excluded": int(args.dental_n),
        },
        "metrics_summary": metrics.summary,
        "topk_marker_ids_selected_mode": topk_marker_ids,
        "topk_marker_ids_rmse_delta": topk_ids_rmse,
        "topk_marker_ids_hypokinesia": topk_ids_hypo,
        "topk_marker_ids_hyperkinesia": topk_ids_hyper,
        "topk_marker_ids_two_sided_amp_dev": topk_ids_twosided,
        "topk_marker_ids_counter_direction": topk_ids_counterdir,
    }
    write_json(out / "summary.json", summary)

    # Plots
    plot_rmse_per_marker(metrics.per_marker, out / "rmse_delta_per_marker.png", top_k=topk)
    plot_rmse_over_time(metrics.per_frame, out / "rmse_delta_over_time.png")

    # Plotly visualization (synchronized): we should also apply time alignment to the reference sequence for visualization.
    # For simplicity, apply the same time alignment to the FULL aligned reference (not only deltas) by reusing lag/dtw mapping on delta descriptors.
    Xref_vis = Xref_aligned.copy()
    if args.time_align == "lag":
        # compute lag from deltas and shift frames
        Dref = delta_sequence(Xref_aligned, neutral_idx_ref, face_idx)
        Dtar = delta_sequence(Xtar_d, neutral_idx_tar, face_idx)
        _, best_lag, _ = apply_lag_alignment(Dref, Dtar, lag_max=args.lag_max)
        warped = np.full_like(Xref_vis, np.nan)
        T = Xref_vis.shape[0]
        if best_lag >= 0:
            warped[0:T-best_lag] = Xref_vis[best_lag:T]
        else:
            lag2 = -best_lag
            warped[lag2:T] = Xref_vis[0:T-lag2]
        Xref_vis = warped
    elif args.time_align == "dtw":
        # dtw on deltas -> map ref frames to each tar frame
        Dref = delta_sequence(Xref_aligned, neutral_idx_ref, face_idx)
        Dtar = delta_sequence(Xtar_d, neutral_idx_tar, face_idx)
        d_ref = global_delta_descriptor(Dref)
        d_tar = global_delta_descriptor(Dtar)
        path, _ = dtw_path(d_ref, d_tar)
        T = Xtar_d.shape[0]
        buckets = {j: [] for j in range(T)}
        for i, j in path:
            if 0 <= j < T:
                buckets[j].append(i)
        warped = np.full_like(Xref_vis, np.nan)
        for j in range(T):
            idxs = buckets.get(j, [])
            if len(idxs) == 0:
                continue
            warped[j] = np.nanmean(Xref_vis[idxs], axis=0)
        Xref_vis = warped

    if go is not None:
        base_title = f"{movement} — Target vs mean healthy (dental frame + neutral facial alignment, time_align={args.time_align})"
        # 1) Selected mode (for backward compatibility)
        make_plotly_overlay(
            Xtar=Xtar_d,
            Xref=Xref_vis,
            face_idx=face_idx,
            topk_marker_ids=topk_marker_ids,
            out_html=out / "overlay_selected_mode.html",
            title=base_title + f" — topk_mode={args.topk_mode}",
        )
        # 2) Most abnormal overall (two-sided amplitude deviation)
        make_plotly_overlay(
            Xtar=Xtar_d,
            Xref=Xref_vis,
            face_idx=face_idx,
            topk_marker_ids=topk_ids_twosided,
            out_html=out / "overlay_topk_twosided.html",
            title=base_title + " — top-K two-sided |log(amp_ratio)|",
        )
        # 3) Hypokinesia (under-movement)
        make_plotly_overlay(
            Xtar=Xtar_d,
            Xref=Xref_vis,
            face_idx=face_idx,
            topk_marker_ids=topk_ids_hypo,
            out_html=out / "overlay_topk_hypokinesia.html",
            title=base_title + " — top-K hypokinesia (lowest amp_ratio)",
        )
        # 4) Hyperkinesia (over-movement)
        make_plotly_overlay(
            Xtar=Xtar_d,
            Xref=Xref_vis,
            face_idx=face_idx,
            topk_marker_ids=topk_ids_hyper,
            out_html=out / "overlay_topk_hyperkinesia.html",
            title=base_title + " — top-K hyperkinesia (highest amp_ratio)",
        
        )
        # 5) Counter-direction markers (directional disagreement vs healthy)
        make_plotly_overlay(
            Xtar=Xtar_d,
            Xref=Xref_vis,
            face_idx=face_idx,
            topk_marker_ids=topk_ids_counterdir,
            out_html=out / "overlay_topk_counter_direction.html",
            title=base_title + " — top-K counter-direction (opp_score)",
        )

    # Also save energy curve for traceability
    energy_df = pd.DataFrame({"frame_raw": np.arange(len(energy), dtype=int), "energy": energy})
    write_csv(out / "energy_curve_raw.csv", energy_df)

    print(f"[OK] Wrote reports to: {out}")

if __name__ == "__main__":
    main()
