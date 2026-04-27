#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics and figures for healthy-evaluation vs pathological kinesia metrics.

Input:
  cohort_metrics_summary.csv from evaluate_kinesia_cohorts.py

Outputs:
  participant_level_metric_table.csv
  cohort_stats_tests.csv
  cohort_metric_boxplots.png
  cohort_metric_effect_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None


DEFAULT_METRICS = [
    "most_abnormal_max_two_sided_amp_dev",
    "hypokinesia_severity",
    "hyperkinesia_max_log_amp_ratio",
    "opposite_direction_max_opp_score",
    "summary_rmse_global_delta",
]

METRIC_LABELS = {
    "most_abnormal_max_two_sided_amp_dev": "Most abnormal\nmax |log amp ratio|",
    "hypokinesia_severity": "Hypokinesia\nseverity",
    "hyperkinesia_max_log_amp_ratio": "Hyperkinesia\nmax log amp ratio",
    "opposite_direction_max_opp_score": "Opposite direction\nmax opp score",
    "summary_rmse_global_delta": "Absolute movement\nRMSE delta",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="cohort_metrics_summary.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for stats and figures.")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--movements", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--healthy_label", default="healthy")
    ap.add_argument("--pathological_label", default="pathological")
    ap.add_argument("--dpi", type=int, default=200)
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()

    valid_idx = np.where(valid)[0]
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid_idx] = q_valid
    return q.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (len(x) * len(y)))


def auroc_from_groups(healthy: np.ndarray, pathological: np.ndarray) -> float:
    if mannwhitneyu is None or len(healthy) == 0 or len(pathological) == 0:
        return np.nan
    try:
        u = mannwhitneyu(pathological, healthy, alternative="two-sided").statistic
        return float(u / (len(pathological) * len(healthy)))
    except Exception:
        return np.nan


def load_and_prepare(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.summary_csv)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    source_metrics = [
        "hypokinesia_min_log_amp_ratio" if metric == "hypokinesia_severity" else metric
        for metric in args.metrics
    ]
    required = {"participant_id", "condition", "movement", *source_metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {missing}")

    df = df[df["movement"].astype(str).isin(args.movements)].copy()
    df = df[df["condition"].astype(str).str.lower().isin([args.healthy_label, args.pathological_label])].copy()
    if "hypokinesia_severity" in args.metrics:
        if "hypokinesia_min_log_amp_ratio" not in df.columns:
            raise ValueError(
                "hypokinesia_severity requires source column hypokinesia_min_log_amp_ratio"
            )
        df["hypokinesia_severity"] = -pd.to_numeric(
            df["hypokinesia_min_log_amp_ratio"], errors="coerce"
        )
    for metric in args.metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    group_cols = ["participant_id", "condition", "movement"]
    participant = (
        df[group_cols + args.metrics]
        .groupby(group_cols, as_index=False)
        .median(numeric_only=True)
    )
    return df, participant


def compute_stats(participant: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric in args.metrics:
        for movement in args.movements:
            sub = participant[participant["movement"] == movement]
            healthy = finite_values(sub[sub["condition"].str.lower() == args.healthy_label][metric])
            patho = finite_values(sub[sub["condition"].str.lower() == args.pathological_label][metric])

            p_value = np.nan
            u_stat = np.nan
            if mannwhitneyu is not None and len(healthy) > 0 and len(patho) > 0:
                res = mannwhitneyu(patho, healthy, alternative="two-sided")
                p_value = float(res.pvalue)
                u_stat = float(res.statistic)

            rows.append({
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "movement": movement,
                "n_healthy": int(len(healthy)),
                "n_pathological": int(len(patho)),
                "healthy_median": float(np.nanmedian(healthy)) if len(healthy) else np.nan,
                "pathological_median": float(np.nanmedian(patho)) if len(patho) else np.nan,
                "delta_median_patho_minus_healthy": (
                    float(np.nanmedian(patho) - np.nanmedian(healthy))
                    if len(healthy) and len(patho) else np.nan
                ),
                "mannwhitney_u": u_stat,
                "p_value": p_value,
                "p_fdr": np.nan,
                "cliffs_delta_patho_vs_healthy": cliffs_delta(patho, healthy),
                "auroc_patho_higher": auroc_from_groups(healthy, patho),
            })

    stats = pd.DataFrame(rows)
    stats["p_fdr"] = benjamini_hochberg(stats["p_value"].tolist())
    return stats


def significance_text(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_boxplots(participant: pd.DataFrame, stats: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    metrics = args.metrics
    movements = args.movements
    n_metrics = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 3.6 * nrows)), squeeze=False)
    axes_flat = axes.ravel()

    healthy_color = "#2F80ED"
    patho_color = "#EB5757"
    rng = np.random.default_rng(7)

    for ax_idx, metric in enumerate(metrics):
        ax = axes_flat[ax_idx]
        positions_h = np.arange(len(movements)) * 3.0 + 1.0
        positions_p = positions_h + 0.8
        data_h = []
        data_p = []

        for movement in movements:
            sub = participant[participant["movement"] == movement]
            data_h.append(finite_values(sub[sub["condition"].str.lower() == args.healthy_label][metric]))
            data_p.append(finite_values(sub[sub["condition"].str.lower() == args.pathological_label][metric]))

        bp_h = ax.boxplot(data_h, positions=positions_h, widths=0.6, patch_artist=True, showfliers=False)
        bp_p = ax.boxplot(data_p, positions=positions_p, widths=0.6, patch_artist=True, showfliers=False)

        for patch in bp_h["boxes"]:
            patch.set_facecolor(healthy_color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(healthy_color)
        for patch in bp_p["boxes"]:
            patch.set_facecolor(patho_color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(patho_color)

        for key in ["whiskers", "caps", "medians"]:
            for artist in bp_h[key]:
                artist.set_color(healthy_color)
            for artist in bp_p[key]:
                artist.set_color(patho_color)

        for i, movement in enumerate(movements):
            xh = positions_h[i] + rng.normal(0, 0.06, size=len(data_h[i]))
            xp = positions_p[i] + rng.normal(0, 0.06, size=len(data_p[i]))
            ax.scatter(xh, data_h[i], s=18, color=healthy_color, alpha=0.65, edgecolors="none")
            ax.scatter(xp, data_p[i], s=18, color=patho_color, alpha=0.65, edgecolors="none")

            stat_row = stats[(stats["metric"] == metric) & (stats["movement"] == movement)]
            if not stat_row.empty:
                label = significance_text(float(stat_row.iloc[0]["p_fdr"]))
                if label:
                    values = np.concatenate([data_h[i], data_p[i]])
                    if len(values) and np.isfinite(values).any():
                        y = np.nanmax(values)
                        pad = 0.06 * max(np.nanmax(values) - np.nanmin(values), 1e-9)
                        ax.text((positions_h[i] + positions_p[i]) / 2, y + pad, label,
                                ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks((positions_h + positions_p) / 2)
        ax.set_xticklabels(movements)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=healthy_color, lw=8, alpha=0.55, label="Healthy eval"),
        plt.Line2D([0], [0], color=patho_color, lw=8, alpha=0.55, label="Pathological"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Healthy Evaluation vs Pathological Kinesia Metrics", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def plot_heatmap(stats: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    matrix = np.full((len(args.metrics), len(args.movements)), np.nan)
    labels = [["" for _ in args.movements] for _ in args.metrics]
    for i, metric in enumerate(args.metrics):
        for j, movement in enumerate(args.movements):
            row = stats[(stats["metric"] == metric) & (stats["movement"] == movement)]
            if row.empty:
                continue
            r = row.iloc[0]
            matrix[i, j] = float(r["cliffs_delta_patho_vs_healthy"])
            ptxt = significance_text(float(r["p_fdr"]))
            labels[i][j] = f"{matrix[i, j]:.2f}\n{ptxt}" if np.isfinite(matrix[i, j]) else ""

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.65 * len(args.metrics))))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(args.movements)))
    ax.set_xticklabels(args.movements)
    ax.set_yticks(np.arange(len(args.metrics)))
    ax.set_yticklabels([METRIC_LABELS.get(m, m).replace("\n", " ") for m in args.metrics])

    for i in range(len(args.metrics)):
        for j in range(len(args.movements)):
            color = "white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 0.55 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cliff's delta: pathological vs healthy")
    ax.set_title("Effect Size Heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sample_level, participant = load_and_prepare(args)
    participant_path = out_dir / "participant_level_metric_table.csv"
    participant.to_csv(participant_path, index=False)

    stats = compute_stats(participant, args)
    stats_path = out_dir / "cohort_stats_tests.csv"
    stats.to_csv(stats_path, index=False)

    plot_boxplots(participant, stats, args, out_dir / "cohort_metric_boxplots.png")
    plot_heatmap(stats, args, out_dir / "cohort_metric_effect_heatmap.png")

    print(f"Sample-level ok rows: {len(sample_level)}")
    print(f"Participant-level rows: {len(participant)}")
    print(f"Wrote: {participant_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {out_dir / 'cohort_metric_boxplots.png'}")
    print(f"Wrote: {out_dir / 'cohort_metric_effect_heatmap.png'}")


if __name__ == "__main__":
    main()
