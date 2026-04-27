#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate FaceMoCap movement samples against healthy reference trajectories.

This is a cohort-level companion to:
  align_mean_movement_refactor/anomaly_kinesia_direction_single_movement_v3.py

It evaluates all selected metadata rows in one process, writes per-sample metric
artifacts, and writes a global CSV suitable for later cohort statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


ANOMALY_SCRIPT = (
    Path(__file__).resolve().parent
    / "align_mean_movement_refactor"
    / "anomaly_kinesia_direction_single_movement_v3.py"
)


def load_anomaly_module():
    spec = importlib.util.spec_from_file_location("akd_v3", ANOMALY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import metric module: {ANOMALY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Metadata CSV with reference_split column.")
    ap.add_argument("--ref_root", required=True, help="Root containing M*/mean_healthy.npy references.")
    ap.add_argument("--out_dir", required=True, help="Output directory for cohort evaluation.")
    ap.add_argument("--root_override", default=None, help="Optional root override for metadata file paths.")
    ap.add_argument("--movements", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--reference_split_col", default="reference_split")
    ap.add_argument("--evaluation_label", default="evaluation")
    ap.add_argument("--conditions", nargs="+", default=["healthy", "pathological"])

    # CSV parsing
    ap.add_argument("--skiprows", type=int, default=5)
    ap.add_argument("--usecols_start", type=int, default=2)
    ap.add_argument("--usecols_end", type=int, default=326, help="End column exclusive. Use 0 for auto to end.")
    ap.add_argument("--dental_n", type=int, default=3)

    # Windowing / resampling
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--neutral_first_pct", type=float, default=0.05)
    ap.add_argument("--energy_thr_percentile", type=float, default=5.0)
    ap.add_argument("--min_window_len", type=int, default=70)
    ap.add_argument("--max_gap", type=int, default=5)

    # Alignment
    ap.add_argument("--trim_frac", type=float, default=0.10)
    ap.add_argument("--huber_iters", type=int, default=3)
    ap.add_argument("--huber_k", type=float, default=1.5)
    ap.add_argument("--try_yaw_flip", action="store_true")
    ap.add_argument("--yaw_flip_axis", default="Z", choices=["X", "Y", "Z"])
    ap.add_argument("--fixed_rot_xyz", nargs=3, type=int, default=[90, 90, 90])

    # Time alignment and metrics
    ap.add_argument("--time_align", default="lag", choices=["none", "lag", "dtw"])
    ap.add_argument("--lag_max", type=int, default=10)
    ap.add_argument("--amp_ref_min_mode", default="percentile", choices=["percentile", "absolute"])
    ap.add_argument("--amp_ref_min_percentile", type=float, default=20.0)
    ap.add_argument("--amp_ref_min", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=10)

    # Outputs
    ap.add_argument("--make_html", action="store_true", help="Also write per-sample Plotly overlays.")
    ap.add_argument("--limit", type=int, default=0, help="Optional debug limit; 0 means all selected rows.")
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def resolve_path(path_value: str, root_override: Optional[str]) -> str:
    path = str(path_value)
    if not root_override or os.path.exists(path):
        return path
    marker = "Data_FaceMoCap"
    if marker in path:
        rel = path.split(marker, 1)[1].lstrip("/\\")
        candidate = os.path.join(root_override, rel)
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(root_override, path.lstrip("/\\"))
    return candidate if os.path.exists(candidate) else path


def select_targets(df: pd.DataFrame, args: argparse.Namespace, ak) -> pd.DataFrame:
    required = {
        "complete_filepath",
        "participant_id",
        "facial_movement",
        "condition",
        "single_movement",
        "valid_for_processing",
        args.reference_split_col,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    out = df.copy()
    out["mov_norm"] = out["facial_movement"].apply(lambda v: safe_normalize_movement(v, ak))
    wanted = {m.upper() for m in args.movements}
    wanted_conditions = {c.strip().lower() for c in args.conditions}

    mask = (
        out["mov_norm"].str.upper().isin(wanted)
        & out["condition"].astype(str).str.strip().str.lower().isin(wanted_conditions)
        & (out[args.reference_split_col].astype(str).str.strip().str.lower() == args.evaluation_label.lower())
        & (pd.to_numeric(out["single_movement"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(out["valid_for_processing"], errors="coerce").fillna(0).astype(int) == 1)
    )
    out = out[mask].copy()
    out = out.sort_values(["condition", "participant_id", "mov_norm", "complete_filepath"])
    if args.limit and args.limit > 0:
        out = out.head(args.limit).copy()
    return out


def safe_normalize_movement(value: object, ak) -> str:
    if pd.isna(value):
        return ""
    try:
        return ak.normalize_movement(value)
    except Exception:
        return ""


def semicolon_ids(df: pd.DataFrame, col: str, top_k: int) -> str:
    if df.empty or col not in df.columns:
        return ""
    return ";".join(df[col].head(top_k).astype(int).astype(str).tolist())


def metric_summary_from_per_marker(pm: pd.DataFrame) -> Dict[str, float]:
    ok = pm[pm["kinesia_ok"] == True].copy()
    return {
        "hypokinesia_min_amp_ratio": float(ok["amp_ratio"].min()) if not ok.empty else np.nan,
        "hypokinesia_min_log_amp_ratio": float(ok["log_amp_ratio"].min()) if not ok.empty else np.nan,
        "hyperkinesia_max_amp_ratio": float(ok["amp_ratio"].max()) if not ok.empty else np.nan,
        "hyperkinesia_max_log_amp_ratio": float(ok["log_amp_ratio"].max()) if not ok.empty else np.nan,
        "most_abnormal_max_two_sided_amp_dev": float(ok["two_sided_amp_dev"].max()) if not ok.empty else np.nan,
        "most_abnormal_median_two_sided_amp_dev": float(ok["two_sided_amp_dev"].median()) if not ok.empty else np.nan,
        "opposite_direction_max_opp_score": float(pm["opp_score"].max()) if not pm.empty else np.nan,
        "opposite_direction_median_opp_score": float(pm["opp_score"].median()) if not pm.empty else np.nan,
        "opposite_direction_max_opp_fraction": float(pm["opp_fraction"].max()) if not pm.empty else np.nan,
        "opposite_direction_median_opp_fraction": float(pm["opp_fraction"].median()) if not pm.empty else np.nan,
        "rmse_delta_max_marker": float(pm["rmse_delta"].max()) if not pm.empty else np.nan,
        "rmse_delta_median_marker": float(pm["rmse_delta"].median()) if not pm.empty else np.nan,
    }


def write_ranked_lists(pm: pd.DataFrame, out: Path, top_k: int) -> Dict[str, str]:
    pm_rmse = pm.sort_values("rmse_delta", ascending=False)
    pm_hypo = pm[pm["kinesia_ok"] == True].sort_values("amp_ratio", ascending=True)
    pm_hyper = pm[pm["kinesia_ok"] == True].sort_values("amp_ratio", ascending=False)
    pm_twosided = pm[pm["kinesia_ok"] == True].sort_values("two_sided_amp_dev", ascending=False)
    pm_counterdir = pm.sort_values("opp_score", ascending=False)

    pm_rmse.head(top_k).to_csv(out / "topk_rmse_delta.csv", index=False)
    pm_hypo.head(top_k).to_csv(out / "topk_hypokinesia.csv", index=False)
    pm_hyper.head(top_k).to_csv(out / "topk_hyperkinesia.csv", index=False)
    pm_twosided.head(top_k).to_csv(out / "topk_two_sided_amp_dev.csv", index=False)
    pm_counterdir.head(top_k).to_csv(out / "topk_counter_direction.csv", index=False)

    return {
        "topk_marker_ids_rmse_delta": semicolon_ids(pm_rmse, "marker_id", top_k),
        "topk_marker_ids_hypokinesia": semicolon_ids(pm_hypo, "marker_id", top_k),
        "topk_marker_ids_hyperkinesia": semicolon_ids(pm_hyper, "marker_id", top_k),
        "topk_marker_ids_two_sided_amp_dev": semicolon_ids(pm_twosided, "marker_id", top_k),
        "topk_marker_ids_counter_direction": semicolon_ids(pm_counterdir, "marker_id", top_k),
    }


def time_aligned_reference_for_html(ak, Xref_aligned: np.ndarray, Xtar_d: np.ndarray, face_idx: np.ndarray, args, neutral_ref: int, neutral_tar: int) -> np.ndarray:
    Xref_vis = Xref_aligned.copy()
    if args.time_align == "lag":
        Dref = ak.delta_sequence(Xref_aligned, neutral_ref, face_idx)
        Dtar = ak.delta_sequence(Xtar_d, neutral_tar, face_idx)
        _, best_lag, _ = ak.apply_lag_alignment(Dref, Dtar, lag_max=args.lag_max)
        warped = np.full_like(Xref_vis, np.nan)
        T = Xref_vis.shape[0]
        if best_lag >= 0:
            warped[0:T - best_lag] = Xref_vis[best_lag:T]
        else:
            lag2 = -best_lag
            warped[lag2:T] = Xref_vis[0:T - lag2]
        return warped
    if args.time_align == "dtw":
        Dref = ak.delta_sequence(Xref_aligned, neutral_ref, face_idx)
        Dtar = ak.delta_sequence(Xtar_d, neutral_tar, face_idx)
        d_ref = ak.global_delta_descriptor(Dref)
        d_tar = ak.global_delta_descriptor(Dtar)
        path, _ = ak.dtw_path(d_ref, d_tar)
        T = Xtar_d.shape[0]
        buckets = {j: [] for j in range(T)}
        for i, j in path:
            if 0 <= j < T:
                buckets[j].append(i)
        warped = np.full_like(Xref_vis, np.nan)
        for j in range(T):
            idxs = buckets.get(j, [])
            if idxs:
                warped[j] = np.nanmean(Xref_vis[idxs], axis=0)
        return warped
    return Xref_vis


def evaluate_one(row: pd.Series, args: argparse.Namespace, ak, out_root: Path) -> Dict[str, object]:
    movement = str(row["mov_norm"])
    target_original = str(row["complete_filepath"])
    target_fp = resolve_path(target_original, args.root_override)
    sid = ak.safe_id_from_filepath(target_original)
    out = out_root / movement / sid
    ensure_dir(out)

    identity = {
        "complete_filepath": target_original,
        "resolved_csv_path": target_fp,
        "movement": movement,
        "metadata_row": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.to_dict().items()},
        "cli_args": vars(args),
    }
    write_json(out / "target_identity.json", identity)

    usecols_end = None if int(args.usecols_end) == 0 else int(args.usecols_end)
    Xtar_w = ak.load_facemocap_csv_points(
        target_fp,
        skiprows=args.skiprows,
        usecols_start=args.usecols_start,
        usecols_end=usecols_end,
    )
    ref_path = Path(args.ref_root) / movement / "mean_healthy.npy"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference mean_healthy.npy not found at: {ref_path}")
    Xref_w = ak.load_mean_healthy_npy(str(ref_path))
    if Xref_w.shape[0] != args.n_frames:
        Xref_w = ak.resample_linear(Xref_w, args.n_frames)
    if Xtar_w.shape[1] != Xref_w.shape[1]:
        raise ValueError(f"Target N={Xtar_w.shape[1]} != Reference N={Xref_w.shape[1]}")

    N = Xtar_w.shape[1]
    if args.dental_n >= N:
        raise ValueError("dental_n must be < total number of markers.")
    face_idx = np.arange(args.dental_n, N, dtype=int)

    neutral_idx_tar_raw = ak.choose_neutral_idx(Xtar_w, face_idx, neutral_first_pct=args.neutral_first_pct)
    start, end, energy = ak.extract_movement_window(
        Xtar_w,
        face_idx,
        neutral_idx_tar_raw,
        energy_thr_percentile=args.energy_thr_percentile,
        min_window_len=args.min_window_len,
        max_gap=args.max_gap,
    )
    Xtar = ak.resample_linear(Xtar_w[start:end], args.n_frames)
    neutral_idx_ref = 0
    neutral_idx_tar = ak.choose_neutral_idx(Xtar, face_idx, neutral_first_pct=args.neutral_first_pct)

    Xtar_d, _, _ = ak.to_dental_coords(Xtar, dental_n=args.dental_n)
    Xref_d, _, _ = ak.to_dental_coords(Xref_w, dental_n=args.dental_n)

    Pref0 = Xref_d[neutral_idx_ref, face_idx, :]
    Qtar0 = Xtar_d[neutral_idx_tar, face_idx, :]
    m0 = ak.valid_mask_points(Pref0) & ak.valid_mask_points(Qtar0)
    if m0.sum() < 3:
        raise RuntimeError(f"Not enough valid facial markers in neutral alignment: {int(m0.sum())}")

    best = ak.best_neutral_face_alignment(
        Pref0[m0],
        Qtar0[m0],
        trim_frac=args.trim_frac,
        huber_iters=args.huber_iters,
        huber_k=args.huber_k,
        fixed_rot_step_xyz=tuple(int(x) for x in args.fixed_rot_xyz),
        try_yaw_flip=bool(args.try_yaw_flip),
        yaw_flip_axis=args.yaw_flip_axis,
    )
    R = best["R"]
    t = best["t"]
    Xref_aligned = np.full_like(Xref_d, np.nan)
    for tt in range(Xref_d.shape[0]):
        P = Xref_d[tt]
        if np.isfinite(P).any():
            Xref_aligned[tt] = (P @ R.T) + t[None, :]

    metrics = ak.compute_metrics_ABC(
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

    metrics.per_marker.to_csv(out / "per_marker_metrics.csv", index=False)
    metrics.per_frame.to_csv(out / "per_frame_metrics.csv", index=False)
    pd.DataFrame({"frame_raw": np.arange(len(energy), dtype=int), "energy": energy}).to_csv(
        out / "energy_curve_raw.csv", index=False
    )
    top_ids = write_ranked_lists(metrics.per_marker, out, args.top_k)

    if args.make_html and getattr(ak, "go", None) is not None:
        Xref_vis = time_aligned_reference_for_html(
            ak, Xref_aligned, Xtar_d, face_idx, args, neutral_idx_ref, neutral_idx_tar
        )
        base_title = f"{movement} - Target vs mean healthy (time_align={args.time_align})"
        for csv_name, html_name, title in [
            ("topk_two_sided_amp_dev.csv", "overlay_topk_twosided.html", "top-K two-sided |log(amp_ratio)|"),
            ("topk_hypokinesia.csv", "overlay_topk_hypokinesia.html", "top-K hypokinesia"),
            ("topk_hyperkinesia.csv", "overlay_topk_hyperkinesia.html", "top-K hyperkinesia"),
            ("topk_counter_direction.csv", "overlay_topk_counter_direction.html", "top-K counter-direction"),
        ]:
            ranked = pd.read_csv(out / csv_name)
            ids = ranked["marker_id"].head(args.top_k).astype(int).tolist() if "marker_id" in ranked.columns else []
            ak.make_plotly_overlay(
                Xtar=Xtar_d,
                Xref=Xref_vis,
                face_idx=face_idx,
                topk_marker_ids=ids,
                out_html=out / html_name,
                title=f"{base_title} - {title}",
            )

    summary = {
        "status": "ok",
        "sample_id": sid,
        "sample_out_dir": str(out),
        "participant_id": row.get("participant_id", ""),
        "condition": row.get("condition", ""),
        "reference_split": row.get(args.reference_split_col, ""),
        "movement": movement,
        "complete_filepath": target_original,
        "resolved_csv_path": target_fp,
        "ref_path": str(ref_path),
        "target_window_raw_start": int(start),
        "target_window_raw_end": int(end),
        "neutral_idx_tar_raw": int(neutral_idx_tar_raw),
        "neutral_idx_tar_resampled": int(neutral_idx_tar),
        "neutral_idx_ref": int(neutral_idx_ref),
        "alignment_score": float(best["score"]),
        "alignment_flip": bool(best["flip"]),
        "alignment_fixed_rot_xyz": str(best.get("fixed_rot_xyz", None)),
        "alignment_inlier_weight_sum": float(best.get("inlier_weight_sum", 0.0)),
    }
    summary.update({f"summary_{k}": v for k, v in metrics.summary.items()})
    summary.update(metric_summary_from_per_marker(metrics.per_marker))
    summary.update(top_ids)
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    ak = load_anomaly_module()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    df = pd.read_csv(args.metadata)
    targets = select_targets(df, args, ak)
    print(f"Selected {len(targets)} evaluation samples")
    if not targets.empty:
        print(
            targets.groupby(["condition", "mov_norm"]).size().rename("n").reset_index().to_string(index=False)
        )

    write_json(out_root / "run_config.json", vars(args))
    targets.to_csv(out_root / "selected_targets.csv", index=False)

    summaries: List[Dict[str, object]] = []
    for idx, (_, row) in enumerate(targets.iterrows(), start=1):
        label = f"{row['condition']} {row['participant_id']} {row['mov_norm']}"
        print(f"[{idx}/{len(targets)}] {label}")
        try:
            summaries.append(evaluate_one(row, args, ak, out_root))
        except Exception as exc:
            err = {
                "status": "fail",
                "participant_id": row.get("participant_id", ""),
                "condition": row.get("condition", ""),
                "reference_split": row.get(args.reference_split_col, ""),
                "movement": row.get("mov_norm", ""),
                "complete_filepath": row.get("complete_filepath", ""),
                "error": str(exc),
            }
            summaries.append(err)
            print(f"  [FAIL] {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_path = out_root / "cohort_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    failures = int((summary_df["status"] == "fail").sum()) if "status" in summary_df.columns else 0
    print(f"\nWrote: {summary_path}")
    print(f"Completed: {len(summary_df) - failures} ok, {failures} failed")


if __name__ == "__main__":
    main()
