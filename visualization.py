from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Heatmap


# ---------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------


def _format_time_label(t: float) -> str:
    """Nicely format time in seconds as h:mm:ss."""
    if t < 60:
        return f"{t:.0f} s"
    m, s = divmod(int(round(t)), 60)
    if m < 60:
        return f"{m:d} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h:d} h {m:02d} min"


# ---------------------------------------------------------------------
# 1D & 2D PLOTS
# ---------------------------------------------------------------------


def make_spatial_profile_figure(
    x: np.ndarray,
    values: np.ndarray,
    name: str,
    units: str,
) -> go.Figure:
    """Profile along river at one instant."""
    x = np.asarray(x)
    values = np.asarray(values)

    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=values,
        mode="lines+markers",
        name=name,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Distance along river (m)",
        yaxis_title=f"{name} ({units})" if units else name,
        title=f"{name} along the river",
    )
    return fig


def make_time_series_figure(
    times: np.ndarray,
    values: np.ndarray,
    x_location: float,
    name: str,
    units: str,
) -> go.Figure:
    """Time series at a fixed location."""
    times = np.asarray(times)
    values = np.asarray(values)

    fig = go.Figure()
    fig.add_scatter(
        x=times,
        y=values,
        mode="lines+markers",
        name=name,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title=f"{name} ({units})" if units else name,
        title=f"{name} at x={x_location:.1f} m",
    )
    return fig


def make_space_time_figure(
    x: np.ndarray,
    times: np.ndarray,
    values_2d: np.ndarray,
    name: str,
    units: str,
) -> go.Figure:
    """
    Space–time heatmap (x vs t).
    """
    x = np.asarray(x)
    times = np.asarray(times)
    values_2d = np.asarray(values_2d)

    fig = go.Figure(
        data=Heatmap(
            x=x,
            y=times,
            z=values_2d,
            colorscale="Viridis",
            colorbar=dict(title=f"{name} ({units})" if units else name),
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Distance along river (m)",
        yaxis_title="Time (s)",
        title=f"{name} – space–time diagram",
    )
    return fig


# ---------------------------------------------------------------------
# TOP VIEW MAP (2D “RIVER” VIEW)
# ---------------------------------------------------------------------


def river_topview_frame(
    x: np.ndarray,
    width: float,
    values: np.ndarray,
    name: str,
    units: str,
) -> go.Figure:
    """
    Single “top view” frame: x on horizontal, width on vertical axis.
    values is the property along x, assumed uniform across width.
    """
    x = np.asarray(x)
    values = np.asarray(values)

    # repeat along width to make a strip
    ny = 2
    y = np.linspace(0.0, width, ny)
    field = np.tile(values[None, :], (ny, 1))

    fig = go.Figure(
        data=Heatmap(
            x=x,
            y=y,
            z=field,
            colorscale="Viridis",
            colorbar=dict(title=f"{name} ({units})" if units else name),
        )
    )

    # grid lines at cell edges
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    edges = np.concatenate(([x[0] - dx / 2.0], x + dx / 2.0))
    for xe in edges:
        fig.add_shape(
            type="line",
            x0=xe,
            x1=xe,
            y0=0.0,
            y1=width,
            line=dict(color="rgba(0,0,0,0.25)", width=1),
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Top view of river – {name}",
        xaxis_title="Distance along river (m)",
        yaxis_title="Width (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_yaxes(showticklabels=False)
    return fig


def river_topview_animation(
    x: np.ndarray,
    width: float,
    values_2d: np.ndarray,
    times: np.ndarray,
    name: str,
    units: str,
    frame_duration_ms: int = 300,
) -> go.Figure:
    """
    Animated top view (time-varying property).
    """
    x = np.asarray(x)
    times = np.asarray(times)
    values_2d = np.asarray(values_2d)

    ny = 2
    y = np.linspace(0.0, width, ny)

    dx = x[1] - x[0] if len(x) > 1 else 1.0
    edges = np.concatenate(([x[0] - dx / 2.0], x + dx / 2.0))

    # only a subset of frames to keep performance reasonable
    n_frames = min(values_2d.shape[0], 60)
    frame_indices = np.linspace(
        0,
        values_2d.shape[0] - 1,
        n_frames,
        dtype=int,
    )

    # initial frame
    idx0 = frame_indices[0]
    field0 = np.tile(values_2d[idx0, :][None, :], (ny, 1))
    heat0 = Heatmap(
        x=x,
        y=y,
        z=field0,
        colorscale="Viridis",
        colorbar=dict(title=f"{name} ({units})" if units else name),
    )

    frames = []
    for idx in frame_indices:
        field = np.tile(values_2d[idx, :][None, :], (ny, 1))
        frames.append(
            go.Frame(
                data=[
                    Heatmap(
                        x=x,
                        y=y,
                        z=field,
                        colorscale="Viridis",
                        colorbar=dict(title=f"{name} ({units})" if units else name),
                    )
                ],
                name=str(idx),
                layout=dict(
                    title_text=(
                        f"Top view of river – {name} "
                        f"(t={_format_time_label(float(times[idx]))})"
                    )
                ),
            )
        )

    fig = go.Figure(data=[heat0], frames=frames)

    # grid lines
    for xe in edges:
        fig.add_shape(
            type="line",
            x0=xe,
            x1=xe,
            y0=0.0,
            y1=width,
            line=dict(color="rgba(0,0,0,0.25)", width=1),
        )

    fig.update_layout(
        title=f"Top view of river – {name} "
              f"(t={_format_time_label(float(times[idx0]))})",
        xaxis_title="Distance along river (m)",
        yaxis_title="Width (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration_ms,
                                           redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                            ),
                        ],
                    ),
                ],
                x=0.02,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(idx)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=_format_time_label(float(times[idx])),
                    )
                    for idx in frame_indices
                ],
                active=0,
                transition=dict(duration=0),
                x=0.15,
                y=1.05,
                currentvalue=dict(prefix="Time: ", visible=True),
            )
        ],
    )

    fig.update_yaxes(showticklabels=False)
    return fig


# ---------------------------------------------------------------------
# DATA EXPORT
# ---------------------------------------------------------------------


def results_to_excel(
    x: np.ndarray,
    times: np.ndarray,
    results: Dict[str, np.ndarray],
) -> bytes:
    """
    Export all properties to one Excel workbook (sheet per property).
    Uses openpyxl as engine (so no xlsxwriter dependency).
    """
    x = np.asarray(x)
    times = np.asarray(times)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, arr in results.items():
            arr = np.asarray(arr)
            df = pd.DataFrame(arr, index=times, columns=x)
            df.index.name = "time_s"
            df.columns.name = "x_m"
            sheet_name = name[:31] if len(name) > 31 else name
            df.to_excel(writer, sheet_name=sheet_name)
    output.seek(0)
    return output.read()


def property_to_csv(
    x: np.ndarray,
    times: np.ndarray,
    values_2d: np.ndarray,
) -> bytes:
    """Export a single property to CSV in long format (time,x,value)."""
    x = np.asarray(x)
    times = np.asarray(times)
    values_2d = np.asarray(values_2d)

    T, N = values_2d.shape
    tt, xx = np.meshgrid(times, x, indexing="ij")
    df = pd.DataFrame(
        {
            "time_s": tt.ravel(),
            "x_m": xx.ravel(),
            "value": values_2d.ravel(),
        }
    )
    return df.to_csv(index=False).encode("utf-8")
