#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code 3b: cohort statistics and figures for robust healthy-envelope metrics.

Input:
  cohort_metrics_summary.csv from evaluate_kinesia_cohorts_2b.py

Outputs:
  participant_level_region_metric_table.csv
  cohort_region_stats_tests.csv
  cohort_region_metric_boxplots.png
  cohort_region_effect_heatmap.png
  exploratory_region_effect_heatmap_*.png/csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None


PRIMARY_METRICS = [
    "region_max_absolute_movement_envelope_severity",
    "region_max_abnormal_amplitude_envelope_severity",
    "region_max_hypokinesia_envelope_severity",
    "region_max_hyperkinesia_envelope_severity",
    "region_max_counter_direction_envelope_severity",
]

METRIC_LABELS = {
    "region_max_absolute_movement_envelope_severity": "Region absolute movement\nabove healthy envelope",
    "region_max_abnormal_amplitude_envelope_severity": "Region amplitude\noutside healthy envelope",
    "region_max_hypokinesia_envelope_severity": "Region hypokinesia\nbelow healthy envelope",
    "region_max_hyperkinesia_envelope_severity": "Region hyperkinesia\nabove healthy envelope",
    "region_max_counter_direction_envelope_severity": "Region counter-direction\nabove healthy envelope",
}

EXPLORATORY_REGION_METRICS = {
    "absolute_movement_envelope": "absolute_movement_envelope_severity_marker_median",
    "abnormal_amplitude_envelope": "abnormal_amplitude_envelope_severity_marker_median",
    "hypokinesia_envelope": "hypokinesia_envelope_severity_marker_median",
    "hyperkinesia_envelope": "hyperkinesia_envelope_severity_marker_median",
    "counter_direction_envelope": "counter_direction_envelope_severity_marker_median",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metrics", nargs="+", default=PRIMARY_METRICS)
    ap.add_argument("--movements", nargs="+", default=["M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--healthy_label", default="healthy")
    ap.add_argument("--pathological_label", default="pathological")
    ap.add_argument("--top_regions", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=200)
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    idx = np.where(valid)[0]
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[idx] = restored
    return q.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (len(x) * len(y)))


def auroc_patho_higher(healthy: np.ndarray, patho: np.ndarray) -> float:
    if mannwhitneyu is None or len(healthy) == 0 or len(patho) == 0:
        return np.nan
    try:
        u = mannwhitneyu(patho, healthy, alternative="two-sided").statistic
        return float(u / (len(patho) * len(healthy)))
    except Exception:
        return np.nan


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


def load_primary_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.summary_csv)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    required = {"participant_id", "condition", "movement", *args.metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Summary CSV missing columns: {missing}")

    df = df[df["movement"].astype(str).isin(args.movements)].copy()
    df = df[df["condition"].astype(str).str.lower().isin([args.healthy_label, args.pathological_label])].copy()
    for metric in args.metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    participant = (
        df[["participant_id", "condition", "movement"] + args.metrics]
        .groupby(["participant_id", "condition", "movement"], as_index=False)
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
            if mannwhitneyu is not None and len(healthy) and len(patho):
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
                "auroc_patho_higher": auroc_patho_higher(healthy, patho),
            })
    stats = pd.DataFrame(rows)
    stats["p_fdr"] = benjamini_hochberg(stats["p_value"].tolist())
    return stats


def plot_primary_boxplots(participant: pd.DataFrame, stats: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    ncols = 2
    nrows = int(np.ceil(len(args.metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 3.6 * nrows)), squeeze=False)
    axes_flat = axes.ravel()
    blue = "#2F80ED"
    red = "#EB5757"
    rng = np.random.default_rng(7)

    for ax_idx, metric in enumerate(args.metrics):
        ax = axes_flat[ax_idx]
        positions_h = np.arange(len(args.movements)) * 3.0 + 1.0
        positions_p = positions_h + 0.8
        data_h = []
        data_p = []
        for movement in args.movements:
            sub = participant[participant["movement"] == movement]
            data_h.append(finite_values(sub[sub["condition"].str.lower() == args.healthy_label][metric]))
            data_p.append(finite_values(sub[sub["condition"].str.lower() == args.pathological_label][metric]))

        bp_h = ax.boxplot(data_h, positions=positions_h, widths=0.6, patch_artist=True, showfliers=False)
        bp_p = ax.boxplot(data_p, positions=positions_p, widths=0.6, patch_artist=True, showfliers=False)
        for patch in bp_h["boxes"]:
            patch.set_facecolor(blue); patch.set_alpha(0.55); patch.set_edgecolor(blue)
        for patch in bp_p["boxes"]:
            patch.set_facecolor(red); patch.set_alpha(0.55); patch.set_edgecolor(red)
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp_h[key]:
                artist.set_color(blue)
            for artist in bp_p[key]:
                artist.set_color(red)

        for i, movement in enumerate(args.movements):
            ax.scatter(positions_h[i] + rng.normal(0, 0.06, len(data_h[i])), data_h[i],
                       s=18, color=blue, alpha=0.65, edgecolors="none")
            ax.scatter(positions_p[i] + rng.normal(0, 0.06, len(data_p[i])), data_p[i],
                       s=18, color=red, alpha=0.65, edgecolors="none")
            row = stats[(stats["metric"] == metric) & (stats["movement"] == movement)]
            if not row.empty:
                sig = significance_text(float(row.iloc[0]["p_fdr"]))
                vals = np.concatenate([data_h[i], data_p[i]])
                if sig and len(vals) and np.isfinite(vals).any():
                    y = np.nanmax(vals)
                    pad = 0.06 * max(np.nanmax(vals) - np.nanmin(vals), 1e-9)
                    ax.text((positions_h[i] + positions_p[i]) / 2, y + pad, sig,
                            ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks((positions_h + positions_p) / 2)
        ax.set_xticklabels(args.movements)
        ax.grid(axis="y", alpha=0.25)

    for idx in range(len(args.metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    handles = [
        plt.Line2D([0], [0], color=blue, lw=8, alpha=0.55, label="Healthy eval"),
        plt.Line2D([0], [0], color=red, lw=8, alpha=0.55, label="Pathological"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Semantic Region Metrics: Healthy Evaluation vs Pathological", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def plot_primary_heatmap(stats: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    matrix = np.full((len(args.metrics), len(args.movements)), np.nan)
    labels = [["" for _ in args.movements] for _ in args.metrics]
    for i, metric in enumerate(args.metrics):
        for j, movement in enumerate(args.movements):
            row = stats[(stats["metric"] == metric) & (stats["movement"] == movement)]
            if row.empty:
                continue
            r = row.iloc[0]
            matrix[i, j] = float(r["cliffs_delta_patho_vs_healthy"])
            labels[i][j] = f"{matrix[i, j]:.2f}\n{significance_text(float(r['p_fdr']))}"

    fig, ax = plt.subplots(figsize=(8.8, max(4.5, 0.7 * len(args.metrics))))
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
    ax.set_title("Primary Region Metric Effect Sizes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def load_region_long(sample_df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    parts = []
    for _, row in sample_df.iterrows():
        p = Path(str(row["sample_out_dir"])) / file_name
        if not p.exists():
            continue
        try:
            reg = pd.read_csv(p)
        except Exception:
            continue
        reg["participant_id"] = row["participant_id"]
        reg["condition"] = row["condition"]
        reg["movement"] = row["movement"]
        reg["sample_id"] = row["sample_id"]
        parts.append(reg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def exploratory_region_stats(region_long: pd.DataFrame, metric_col: str, group_cols: List[str], args: argparse.Namespace) -> pd.DataFrame:
    if region_long.empty or metric_col not in region_long.columns:
        return pd.DataFrame()
    cols = ["participant_id", "condition", "movement"] + group_cols + [metric_col]
    data = region_long[cols].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    participant = (
        data.groupby(["participant_id", "condition", "movement"] + group_cols, as_index=False)
        .median(numeric_only=True)
    )

    rows = []
    for keys, sub in participant.groupby(group_cols + ["movement"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {col: val for col, val in zip(group_cols + ["movement"], keys)}
        healthy = finite_values(sub[sub["condition"].str.lower() == args.healthy_label][metric_col])
        patho = finite_values(sub[sub["condition"].str.lower() == args.pathological_label][metric_col])
        p_value = np.nan
        if mannwhitneyu is not None and len(healthy) and len(patho):
            p_value = float(mannwhitneyu(patho, healthy, alternative="two-sided").pvalue)
        rows.append({
            **key_map,
            "metric_col": metric_col,
            "n_healthy": int(len(healthy)),
            "n_pathological": int(len(patho)),
            "healthy_median": float(np.nanmedian(healthy)) if len(healthy) else np.nan,
            "pathological_median": float(np.nanmedian(patho)) if len(patho) else np.nan,
            "delta_median_patho_minus_healthy": (
                float(np.nanmedian(patho) - np.nanmedian(healthy))
                if len(healthy) and len(patho) else np.nan
            ),
            "p_value": p_value,
            "p_fdr": np.nan,
            "cliffs_delta_patho_vs_healthy": cliffs_delta(patho, healthy),
            "auroc_patho_higher": auroc_patho_higher(healthy, patho),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = benjamini_hochberg(out["p_value"].tolist())
    return out


def plot_exploratory_heatmap(stats: pd.DataFrame, label_col: str, title: str, top_n: int, out_path: Path, dpi: int) -> None:
    if stats.empty:
        return
    score = (
        stats.groupby(label_col)["cliffs_delta_patho_vs_healthy"]
        .apply(lambda s: float(np.nanmax(np.abs(s))))
        .sort_values(ascending=False)
    )
    labels = score.head(top_n).index.tolist()
    movements = sorted(stats["movement"].dropna().unique().tolist())
    matrix = np.full((len(labels), len(movements)), np.nan)
    text = [["" for _ in movements] for _ in labels]
    for i, label in enumerate(labels):
        for j, movement in enumerate(movements):
            row = stats[(stats[label_col] == label) & (stats["movement"] == movement)]
            if row.empty:
                continue
            r = row.iloc[0]
            matrix[i, j] = float(r["cliffs_delta_patho_vs_healthy"])
            text[i][j] = f"{matrix[i, j]:.2f}\n{significance_text(float(r['p_fdr']))}"

    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(labels))))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(movements)))
    ax.set_xticklabels(movements)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(movements)):
            color = "white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 0.55 else "black"
            ax.text(j, i, text[i][j], ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cliff's delta: pathological vs healthy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sample_level, participant = load_primary_tables(args)
    participant_path = out_dir / "participant_level_region_metric_table.csv"
    participant.to_csv(participant_path, index=False)

    stats = compute_stats(participant, args)
    stats_path = out_dir / "cohort_region_stats_tests.csv"
    stats.to_csv(stats_path, index=False)
    plot_primary_boxplots(participant, stats, args, out_dir / "cohort_region_metric_boxplots.png")
    plot_primary_heatmap(stats, args, out_dir / "cohort_region_effect_heatmap.png")

    region_long = load_region_long(sample_level, "per_region_metrics.csv")
    region_side_long = load_region_long(sample_level, "per_region_side_metrics.csv")
    if not region_long.empty:
        region_long.to_csv(out_dir / "region_metric_long_table.csv", index=False)
    if not region_side_long.empty:
        region_side_long["region_side"] = region_side_long["region"].astype(str) + " | " + region_side_long["side"].astype(str)
        region_side_long.to_csv(out_dir / "region_side_metric_long_table.csv", index=False)

    for short_name, metric_col in EXPLORATORY_REGION_METRICS.items():
        reg_stats = exploratory_region_stats(region_long, metric_col, ["region"], args)
        if not reg_stats.empty:
            csv_path = out_dir / f"exploratory_region_stats_{short_name}.csv"
            reg_stats.to_csv(csv_path, index=False)
            plot_exploratory_heatmap(
                reg_stats,
                label_col="region",
                title=f"Exploratory Region Effects: {short_name.replace('_', ' ')}",
                top_n=args.top_regions,
                out_path=out_dir / f"exploratory_region_effect_heatmap_{short_name}.png",
                dpi=args.dpi,
            )

        rs_stats = exploratory_region_stats(region_side_long, metric_col, ["region_side"], args)
        if not rs_stats.empty:
            csv_path = out_dir / f"exploratory_region_side_stats_{short_name}.csv"
            rs_stats.to_csv(csv_path, index=False)
            plot_exploratory_heatmap(
                rs_stats,
                label_col="region_side",
                title=f"Exploratory Region-Side Effects: {short_name.replace('_', ' ')}",
                top_n=args.top_regions,
                out_path=out_dir / f"exploratory_region_side_effect_heatmap_{short_name}.png",
                dpi=args.dpi,
            )

    print(f"Sample-level ok rows: {len(sample_level)}")
    print(f"Participant-level rows: {len(participant)}")
    print(f"Wrote: {participant_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {out_dir / 'cohort_region_metric_boxplots.png'}")
    print(f"Wrote: {out_dir / 'cohort_region_effect_heatmap.png'}")


if __name__ == "__main__":
    main()
