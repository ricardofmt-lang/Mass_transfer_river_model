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
        # fallback: just first time
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
    x_location: user-chosen coordinate (m)
    x_grid: array of cell centers (m)
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


# ------------------------ 3D VISUALISATIONS ----------------------------------


def river_surface_figure(x, width, arr, times, t_index, pname, units, n_cross=8):
    """
    3D view of the river surface.
    Geometry: 1D channel extruded across width, with a flat water surface (z=0).
    Colour of the surface encodes the property (arr[t_index, :]).
    """
    x = np.asarray(x)
    times = np.asarray(times)
    C = arr[t_index, :]  # shape (n_x,)
    y = np.linspace(0.0, width, n_cross)

    X, Y = np.meshgrid(x, y)          # shapes (n_cross, n_x)
    Z = np.zeros_like(X)              # flat water surface
    C2d = np.tile(C, (n_cross, 1))    # same conc along width

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=C2d,
        colorscale="Viridis",
        colorbar=dict(title=f"{pname} ({units})"),
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"{pname} on river surface at t = {times[t_index]:.1f} s",
        scene=dict(
            xaxis_title="Distance along river (m)",
            yaxis_title="River width (m)",
            zaxis_title="Water surface",
            zaxis=dict(showticklabels=False),
        ),
    )
    fig.update_traces(showscale=True)
    return fig
