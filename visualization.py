import io
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _format_time_label(t: float) -> str:
    """Pretty formatting of time for titles/labels."""
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.1f} h"


# ---------------------------------------------------------------------
# 1D PLOTS
# ---------------------------------------------------------------------


def make_spatial_profile_figure(
    x: np.ndarray,
    values: np.ndarray,
    t: float,
    name: str,
    units: str,
) -> go.Figure:
    """C(x) at a single time."""
    x = np.asarray(x)
    values = np.asarray(values)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines+markers",
            line=dict(width=2),
            name=name,
        )
    )
    fig.update_layout(
        title=f"{name} along river at t={_format_time_label(t)}",
        xaxis_title="Distance along river (m)",
        yaxis_title=f"{name} ({units})" if units else name,
        template="plotly_white",
    )
    return fig


def make_time_series_figure(
    times: np.ndarray,
    values: np.ndarray,
    x_loc: float,
    name: str,
    units: str,
) -> go.Figure:
    """C(t) at one spatial position."""
    times = np.asarray(times)
    values = np.asarray(values)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines+markers",
            line=dict(width=2),
            name=name,
        )
    )
    fig.update_layout(
        title=f"{name} at x={x_loc:.1f} m",
        xaxis_title="Time (s)",
        yaxis_title=f"{name} ({units})" if units else name,
        template="plotly_white",
    )
    return fig


def make_space_time_figure(
    times: np.ndarray,
    x: np.ndarray,
    values_2d: np.ndarray,
    name: str,
    units: str,
) -> go.Figure:
    """C(x,t) as a space–time heatmap."""
    times = np.asarray(times)
    x = np.asarray(x)
    values_2d = np.asarray(values_2d)

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=times,
            z=values_2d,
            colorscale="Viridis",
            colorbar=dict(title=f"{name} ({units})" if units else name),
        )
    )
    fig.update_layout(
        title=f"{name} – space–time map",
        xaxis_title="Distance along river (m)",
        yaxis_title="Time (s)",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------
# 2D TOP-VIEW RIVER MAP
# ---------------------------------------------------------------------


def river_topview_frame(
    x: np.ndarray,
    width: float,
    values: np.ndarray,
    name: str,
    units: str,
    n_width_points: int = 25,
) -> go.Figure:
    """
    Static 2D top view: x-axis = distance along river, y-axis = width.
    Values are constant across width (1D field extruded laterally).
    """
    x = np.asarray(x)
    values = np.asarray(values)

    # lateral coordinate
    y = np.linspace(0.0, width, n_width_points)
    field = np.tile(values, (n_width_points, 1))

    # cell edges (for grid lines)
    if len(x) > 1:
        dx = x[1] - x[0]
    else:
        dx = 1.0
    edges = np.concatenate(
        (
            [x[0] - dx / 2.0],
            (x[:-1] + x[1:]) / 2.0,
            [x[-1] + dx / 2.0],
        )
    )

    heat = go.Heatmap(
        x=x,
        y=y,
        z=field,
        colorscale="Viridis",
        colorbar=dict(title=f"{name} ({units})" if units else name),
    )

    fig = go.Figure(data=[heat])

    # vertical grid lines for each cell
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
        title=f"Top view of river – {name}",
        xaxis_title="Distance along river (m)",
        yaxis_title="Width (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    # width is not a “variable”, so hide ticks
    fig.update_yaxes(showticklabels=False)
    return fig


def river_topview_animation(
    x: np.ndarray,
    width: float,
    values_2d: np.ndarray,
    times: np.ndarray,
    name: str,
    units: str,
    frame_duration_ms: int = 200,
    max_frames: int = 80,
    n_width_points: int = 25,
) -> go.Figure:
    """
    Animated 2D top view. We sample at most `max_frames` time steps.
    """
    x = np.asarray(x)
    values_2d = np.asarray(values_2d)
    times = np.asarray(times)

    n_times, n_cells = values_2d.shape
    assert n_cells == x.size

    if n_times <= max_frames:
        frame_indices = np.arange(n_times)
    else:
        frame_indices = np.linspace(0, n_times - 1, max_frames).astype(int)

    y = np.linspace(0.0, width, n_width_points)

    if len(x) > 1:
        dx = x[1] - x[0]
    else:
        dx = 1.0
    edges = np.concatenate(
        (
            [x[0] - dx / 2.0],
            (x[:-1] + x[1:]) / 2.0,
            [x[-1] + dx / 2.0],
        )
    )

    # for colour scale consistency
    vmin = float(np.nanmin(values_2d))
    vmax = float(np.nanmax(values_2d))

    # initial frame
    idx0 = frame_indices[0]
    field0 = np.tile(values_2d[idx0, :], (n_width_points, 1))

    heat0 = go.Heatmap(
        x=x,
        y=y,
        z=field0,
        colorscale="Viridis",
        colorbar=dict(title=f"{name} ({units})" if units else name),
        zmin=vmin,
        zmax=vmax,
    )

    frames = []
    for idx in frame_indices:
        field = np.tile(values_2d[idx, :], (n_width_points, 1))
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=field,
                        colorscale="Viridis",
                        colorbar=dict(title=f"{name} ({units})" if units else name),
                        zmin=vmin,
                        zmax=vmax,
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
    """Export all properties to one Excel workbook (sheet per property)."""
    x = np.asarray(x)
    times = np.asarray(times)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
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
