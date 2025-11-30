import numpy as np
import pandas as pd
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go


def make_excel_workbook(times, x, results, prop_cfg):
    """
    Create a single Excel workbook with one sheet per property.
    Each sheet: columns = [time (s), x0, x1, ..., xN].
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for pname, arr in results.items():
            df = pd.DataFrame(arr, columns=x)
            df.insert(0, "time (s)", times)
            sheet_name = pname[:31] or "Prop"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf


# ------------------------ 2D VISUALISATIONS ----------------------------------


def profiles_over_space_figure(x, times, arr, pname, units, time_indices):
    """
    Multiple profiles C(x) at selected output times.
    arr shape: (n_times, n_x)
    time_indices: list of indices into 'times'.
    """
    x = np.asarray(x)
    times = np.asarray(times)
    data_frames = []
    for idx in time_indices:
        if 0 <= idx < arr.shape[0]:
            df = pd.DataFrame(
                {
                    "Distance (m)": x,
                    "Value": arr[idx, :],
                    "Time": f"{times[idx]:.1f} s",
                }
            )
            data_frames.append(df)
    if not data_frames:
        df = pd.DataFrame(
            {
                "Distance (m)": x,
                "Value": arr[0, :],
                "Time": f"{times[0]:.1f} s",
            }
        )
        data_frames.append(df)

    df_all = pd.concat(data_frames, ignore_index=True)
    fig = px.line(
        df_all,
        x="Distance (m)",
        y="Value",
        color="Time",
        labels={"Value": f"{pname} ({units})"},
    )
    fig.update_layout(
        title=f"{pname} profiles at selected times",
        legend_title="Time",
    )
    return fig


def timeseries_figure(times, arr, pname, units, x_location, x_grid):
    """
    Time series C(t) at a chosen spatial location x.
    """
    x_grid = np.asarray(x_grid)
    times = np.asarray(times)
    idx = int(np.argmin(np.abs(x_grid - x_location)))
    x_near = x_grid[idx]
    df = pd.DataFrame(
        {
            "Time (s)": times,
            "Value": arr[:, idx],
        }
    )
    fig = px.line(
        df,
        x="Time (s)",
        y="Value",
        labels={"Value": f"{pname} ({units})"},
    )
    fig.update_layout(
        title=f"{pname} time series at x ≈ {x_near:.1f} m",
    )
    return fig, x_near


def curtain_figure(x, times, arr, pname, units):
    """
    Space-time "curtain" plot: C(x, t) as a heatmap.
    arr shape: (n_times, n_x)
    """
    x = np.asarray(x)
    times = np.asarray(times)
    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=times,
            z=arr,
            colorscale="Viridis",
            colorbar=dict(title=f"{pname} ({units})"),
        )
    )
    fig.update_layout(
        xaxis_title="Distance along river (m)",
        yaxis_title="Time (s)",
        title=f"{pname} space–time evolution",
    )
    return fig


# ------------------------ helpers for 3D -------------------------------------


def _compute_bed_and_water(x, width, depth, slope, n_cross):
    """
    Build bed and water surfaces:
      - bed has slope (m/m)
      - water surface at z = 0
    """
    x = np.asarray(x)
    y = np.linspace(0.0, width, n_cross)
    X, Y = np.meshgrid(x, y)

    # bed elevation: upstream around -depth, then going down with slope·x
    bed_line = -depth - slope * x
    bed_z = np.tile(bed_line, (n_cross, 1))
    water_z = np.zeros_like(X)

    return X, Y, bed_z, water_z


def _cell_grid_traces(x, width, water_level=0.0):
    """
    Draw vertical grid lines at cell boundaries on the water surface.
    """
    x = np.asarray(x)
    if len(x) < 2:
        return []

    dx = float(np.mean(np.diff(x)))
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - dx / 2
    edges[-1] = x[-1] + dx / 2

    traces = []
    for xb in edges:
        traces.append(
            go.Scatter3d(
                x=[xb, xb],
                y=[0.0, width],
                z=[water_level, water_level],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.5)", width=2),
                showlegend=False,
            )
        )
    return traces


# ------------------------ 3D VISUALISATIONS ----------------------------------


def river_surface_figure(x, width, depth, slope, arr, times, t_index, pname, units, n_cross=4):
    """
    Static 3D view of the river longitudinal profile:
      - bed with slope
      - water surface at z = 0, coloured by property at given time
      - grid lines marking cells along x
    """
    x = np.asarray(x)
    times = np.asarray(times)
    X, Y, bed_z, water_z = _compute_bed_and_water(x, width, depth, slope, n_cross)

    C = arr[t_index, :]
    C2d = np.tile(C, (n_cross, 1))

    bed = go.Surface(
        x=X,
        y=Y,
        z=bed_z,
        colorscale="Greys",
        showscale=False,
        opacity=0.4,
    )
    water = go.Surface(
        x=X,
        y=Y,
        z=water_z,
        surfacecolor=C2d,
        colorscale="Viridis",
        colorbar=dict(title=f"{pname} ({units})"),
    )

    grid_traces = _cell_grid_traces(x, width)

    fig = go.Figure(data=[bed, water] + grid_traces)
    fig.update_layout(
        title=f"{pname} on river surface at t = {times[t_index]:.1f} s",
        scene=dict(
            xaxis_title="Distance along river (m)",
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            zaxis_title="Elevation (m)",
            aspectmode="data",
        ),
    )
    # camera looking at river profile (side-ish)
    fig.update_layout(
        scene_camera=dict(eye=dict(x=1.6, y=0.3, z=0.8))
    )
    return fig


def river_surface_animation_figure(
    x,
    width,
    depth,
    slope,
    arr,
    times,
    pname,
    units,
    n_cross=4,
    max_frames=60,
    frame_duration_ms=150,
):
    """
    Animated 3D view of the river profile over time.
    Play/pause + slider, with adjustable frame duration.
    """
    x = np.asarray(x)
    times = np.asarray(times)
    n_times = arr.shape[0]

    X, Y, bed_z, water_z = _compute_bed_and_water(x, width, depth, slope, n_cross)

    if n_times <= max_frames:
        frame_indices = np.arange(n_times)
    else:
        frame_indices = np.linspace(0, n_times - 1, max_frames).astype(int)

    # initial frame
    C0 = arr[frame_indices[0], :]
    C0_2d = np.tile(C0, (n_cross, 1))

    bed = go.Surface(
        x=X,
        y=Y,
        z=bed_z,
        colorscale="Greys",
        showscale=False,
        opacity=0.4,
    )
    water0 = go.Surface(
        x=X,
        y=Y,
        z=water_z,
        surfacecolor=C0_2d,
        colorscale="Viridis",
        colorbar=dict(title=f"{pname} ({units})"),
    )

    grid_traces = _cell_grid_traces(x, width)

    frames = []
    for idx in frame_indices:
        C = arr[idx, :]
        C2d = np.tile(C, (n_cross, 1))
        frame = go.Frame(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=bed_z,
                    colorscale="Greys",
                    showscale=False,
                    opacity=0.4,
                ),
                go.Surface(
                    x=X,
                    y=Y,
                    z=water_z,
                    surfacecolor=C2d,
                    colorscale="Viridis",
                    colorbar=dict(title=f"{pname} ({units})"),
                ),
            ],
            name=str(idx),
        )
        frames.append(frame)

    fig = go.Figure(data=[bed, water0] + grid_traces, frames=frames)

    fig.update_layout(
        title=f"{pname} animation along river",
        scene=dict(
            xaxis_title="Distance along river (m)",
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            zaxis_title="Elevation (m)",
            aspectmode="data",
        ),
        scene_camera=dict(eye=dict(x=1.6, y=0.3, z=0.8)),
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
                                frame=dict(duration=frame_duration_ms, redraw=True),
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
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                x=0.1,
                y=0,
                xanchor="right",
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
                        label=f"{times[idx]:.1f}",
                    )
                    for idx in frame_indices
                ],
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=50),
                len=0.9,
            )
        ],
    )
    return fig
