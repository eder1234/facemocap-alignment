#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for align-and-mean movement pipeline.

This module keeps the same CLI flags and behavior as the original monolithic script.
"""

from __future__ import annotations

# stdlib
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io.metadata import infer_cols, normalize_movement, is_healthy_from_condition
from .io.outputs import ensure_dir
from .io.facemocap_csv import load_facemocap_csv
from .geometry.dental_frame import compute_dental_frame, world_to_dental, dental_marker_drift
from .processing.neutral import pick_neutral_frame_most_complete
from .geometry.rotations import fixed_rot_xyz
from .geometry.rigid_alignment import (
    trimmed_ransac_rigid, maybe_yaw_flip, huber_irls_refine, apply_rt,
    kabsch, rms, trimmed_indices
)
from .processing.movement_window import displacement_energy, extract_active_window
from .processing.resampling import resample_sequence_nan_robust
from .viz.plotly_anim import make_animation_html

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=str)
    ap.add_argument("--template_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--root_override", default=None, type=str)
    ap.add_argument("--movements", nargs="+", default=["M1","M2","M3","M4","M5"])
    ap.add_argument("--neutral_first_pct", type=float, default=0.05)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--energy_thr_percentile", type=float, default=70.0)
    ap.add_argument("--min_window_len", type=int, default=10)

    ap.add_argument("--min_points", type=int, default=60)

    ap.add_argument("--ransac", action="store_true")
    ap.add_argument("--ransac_trials", type=int, default=4000)
    ap.add_argument("--ransac_subset", type=int, default=4)
    ap.add_argument("--trim_frac", type=float, default=0.10)

    ap.add_argument("--try_yaw_flip", action="store_true")
    ap.add_argument("--yaw_flip_axis", type=str, default="Z")

    ap.add_argument("--huber_iters", type=int, default=3)
    ap.add_argument("--huber_k", type=float, default=1.5)

    ap.add_argument("--fixed_rot_xyz", nargs=3, type=float, default=[90.0, 90.0, 90.0])

    ap.add_argument("--n_show", type=int, default=10)
    ap.add_argument("--export_sample_anims", type=int, default=3, help="Per movement, export this many sample animations in addition to mean.")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Parse metadata
    df = pd.read_csv(args.metadata)
    path_col, mov_col, grp_col = infer_cols(df)

    # optional filters
    if "valid_for_processing" in df.columns:
        df = df[df["valid_for_processing"].astype(int) == 1].copy()
    if "single_movement" in df.columns:
        df = df[df["single_movement"].astype(int) == 1].copy()

    df["mov_norm"] = df[mov_col].apply(normalize_movement)
    wanted = set([m.upper() for m in args.movements])
    df = df[df["mov_norm"].str.upper().isin(wanted)].copy()

    # Resolve paths
    def resolve_path(p: str) -> Path:
        pth = Path(str(p))
        if pth.is_absolute():
            return pth
        if args.root_override:
            return Path(args.root_override) / pth
        return pth

    df["csv_path"] = df[path_col].apply(resolve_path)

    # Template neutral: frame 0 dental + fixed rot
    template_path = resolve_path(args.template_csv)
    template_frames = load_facemocap_csv(template_path)
    template_frame0 = template_frames[0:1]  # (1,108,3)
    template_dental, ok = world_to_dental(template_frame0)
    if not ok[0]:
        raise RuntimeError(f"Template dental frame invalid at frame 0: {template_path}")
    Rfix = fixed_rot_xyz(tuple(args.fixed_rot_xyz))
    template_dental0 = (Rfix @ template_dental[0].T).T  # 108x3
    template_face = template_dental0[3:, :]  # 105x3

    # Process per sample
    rng = np.random.default_rng(args.seed)
    facial_slice = slice(3, 108)

    rows = []
    # Store per-movement processed sequences
    by_mov: Dict[str, List[Dict]] = {m: [] for m in wanted}

    for _, r in df.iterrows():
        csv_path = Path(r["csv_path"])
        mov = str(r["mov_norm"]).upper()
        cond = r[grp_col]
        is_healthy = (str(cond).strip().lower() == "healthy")

        rec = dict(mov=mov, csv=str(csv_path), is_healthy=is_healthy)
        try:
            frames_world = load_facemocap_csv(csv_path)
            frames_dental, ok_d = world_to_dental(frames_world)
            rec["dental_ok_frac"] = float(ok_d.mean())
            rec["dental_drift_rms"] = dental_marker_drift(frames_dental)

            neutral_idx = pick_neutral_frame_most_complete(frames_dental, args.neutral_first_pct, facial_slice)
            rec["neutral_idx"] = int(neutral_idx)

            # Apply fixed rot to ALL frames (canonicalize)
            frames_can = (Rfix @ frames_dental.reshape(-1,3).T).T.reshape(frames_dental.shape)

            # Build correspondences for neutral facial points
            Pn = frames_can[neutral_idx, 3:, :]  # 105x3
            Qn = template_face  # 105x3

            valid = np.isfinite(Pn).all(axis=1) & np.isfinite(Qn).all(axis=1)
            rec["n_valid_face"] = int(valid.sum())
            if valid.sum() < args.min_points:
                raise ValueError(f"Not enough valid facial correspondences: {valid.sum()} < {args.min_points}")

            P = Pn[valid]
            Q = Qn[valid]

            if args.ransac:
                R, t, inl, score = trimmed_ransac_rigid(
                    P, Q, trials=args.ransac_trials, subset=args.ransac_subset,
                    trim_frac=args.trim_frac, rng=rng
                )
                rec["trimmed_rms_fit"] = float(score)
                rec["n_inliers"] = int(inl.sum())
                if args.try_yaw_flip:
                    R, t, used_flip, score2 = maybe_yaw_flip(P, Q, R, t, axis=args.yaw_flip_axis)
                    rec["used_yaw_flip"] = bool(used_flip)
                    rec["rms_after_flip"] = float(score2)
                if args.huber_iters > 0:
                    # Recompute residuals and inliers after potential flip
                    res = np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)
                    inl2 = trimmed_indices(res, args.trim_frac)
                    R, t = huber_irls_refine(P, Q, R, t, inl2, iters=args.huber_iters, k=args.huber_k)
            else:
                R, t = kabsch(P, Q)
                rec["trimmed_rms_fit"] = float(rms(np.linalg.norm(apply_rt(P, R, t) - Q, axis=1)))
                rec["n_inliers"] = int(P.shape[0])
                rec["used_yaw_flip"] = False

            # Apply to all frames (facial + dental + everything)
            Tn = frames_can.shape[0]
            flat = frames_can.reshape(Tn*108, 3)
            flat2 = apply_rt(flat, R, t)
            frames_aligned = flat2.reshape(Tn, 108, 3)

            # Movement window by energy from neutral
            E = displacement_energy(frames_aligned, neutral_idx, facial_slice)
            w0, w1 = extract_active_window(E, neutral_idx, thr_percentile=args.energy_thr_percentile, min_len=args.min_window_len)
            rec["win_start"] = int(w0)
            rec["win_end"] = int(w1)

            # Resample to fixed length
            frames_res = resample_sequence_nan_robust(frames_aligned, w0, w1, args.n_frames)
            rec["n_frames_in"] = int(frames_world.shape[0])
            rec["n_frames_win"] = int(w1 - w0 + 1)

            # store
            by_mov[mov].append(dict(
                csv=str(csv_path),
                is_healthy=is_healthy,
                frames_res=frames_res,
                neutral_idx=int(neutral_idx),
                win=(int(w0), int(w1)),
                trimmed_rms=float(rec.get("trimmed_rms_fit", np.nan)),
                n_valid=int(rec.get("n_valid_face", 0)),
            ))

            rec["status"] = "ok"
            rec["error"] = ""
        except Exception as e:
            rec["status"] = "fail"
            rec["error"] = str(e)
        rows.append(rec)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary_sequences.csv"
    summary.to_csv(summary_path, index=False)

    # Compute per-movement healthy mean and export animations
    for mov, items in by_mov.items():
        if len(items) == 0:
            continue
        mov_dir = out_dir / mov
        ensure_dir(mov_dir)
        ensure_dir(mov_dir / "diagnostics")

        # Separate healthy sequences for mean
        healthy = [it["frames_res"] for it in items if it["is_healthy"]]
        if len(healthy) == 0:
            # still export a mean over all as fallback
            healthy = [it["frames_res"] for it in items]

        stack = np.stack(healthy, axis=0)  # N,T,108,3
        mean_seq = np.nanmean(stack, axis=0)  # T,108,3

        np.save(mov_dir / "mean_healthy.npy", mean_seq)
        # also CSV: one row per frame, flattened
        mean_flat = mean_seq.reshape(args.n_frames, 108*3)
        mean_csv = mov_dir / "mean_healthy.csv"
        pd.DataFrame(mean_flat).to_csv(mean_csv, index=False, header=False)

        # Interactive animation for mean
        make_animation_html(mean_seq, mov_dir / "mean_healthy_animation.html", title=f"{mov} mean healthy (aligned, no head motion)", facial_only=True)

        # Optionally export a few sample animations
        n_export = min(args.export_sample_anims, len(items))
        for k in range(n_export):
            it = items[k]
            sid = Path(it["csv"]).stem
            make_animation_html(it["frames_res"], mov_dir / f"sample_animation_{sid}.html",
                                title=f"{mov} sample {sid} (aligned, resampled)", facial_only=True)

    # Also export template neutral visualization
    make_animation_html(template_dental0[None, :, :], out_dir / "template_neutral_frame0.html",
                        title="Template neutral (frame 0) in its dental frame (canonicalized)", facial_only=True)

    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote per-movement outputs under: {out_dir}")
