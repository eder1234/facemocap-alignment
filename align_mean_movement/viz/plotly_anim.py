from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

def make_animation_html(points_seq: np.ndarray, out_html: Path, title: str, facial_only: bool = True, marker_size: int = 3) -> None:
    """
    Create interactive 3D animation (slider + play) and save as HTML.
    points_seq: (T,108,3) or (T,105,3) if facial_only already.
    """
    seq = points_seq.copy()
    if facial_only and seq.shape[1] == 108:
        seq = seq[:, 3:, :]  # facial only

    # Determine axis ranges from finite points
    finite = np.isfinite(seq).all(axis=-1)
    pts = seq[finite]
    if pts.size == 0:
        # produce empty
        fig = go.Figure()
        fig.update_layout(title=title)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        return

    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    pad = 0.05 * np.max(maxs - mins)
    xr = [float(mins[0]-pad), float(maxs[0]+pad)]
    yr = [float(mins[1]-pad), float(maxs[1]+pad)]
    zr = [float(mins[2]-pad), float(maxs[2]+pad)]

    frames = []
    for i in range(seq.shape[0]):
        P = seq[i]
        ok = np.isfinite(P).all(axis=1)
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=P[ok,0], y=P[ok,1], z=P[ok,2],
                mode="markers",
                marker=dict(size=marker_size),
            )],
            name=str(i)
        ))

    # initial
    P0 = seq[0]
    ok0 = np.isfinite(P0).all(axis=1)
    fig = go.Figure(
        data=[go.Scatter3d(
            x=P0[ok0,0], y=P0[ok0,1], z=P0[ok0,2],
            mode="markers",
            marker=dict(size=marker_size),
        )],
        frames=frames
    )

    # slider
    steps = []
    for i in range(seq.shape[0]):
        steps.append(dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=str(i)
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=xr),
            yaxis=dict(range=yr),
            zaxis=dict(range=zr),
            aspectmode="data",
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Frame: "),
            pad=dict(t=50),
            steps=steps,
        )]
    )

    fig.write_html(str(out_html), include_plotlyjs="cdn")


# ----------------------------
# Metadata parsing
# ----------------------------
