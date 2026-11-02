"""Animated 3D view of the one-dimensional river model's printed values.

Each model cell is a colored slice across the river's whole cross-section.
The top, banks, and end faces form a channel with the entered width and depth.
The model does not resolve variation across the width or through the depth.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .numerical import SimulationResult


COLORSCALE = "RdYlBu_r"  # low values blue, high values red


# Cell centres define the physical x positions; edge reconstruction closes the first and last cells.
def _cell_edges(centers: list[float], length_m: float) -> list[float]:
    if not centers:
        raise ValueError("A 3D river requires at least one cell")
    if len(centers) == 1:
        half = length_m / 2
        return [centers[0] - half, centers[0] + half]
    return ([centers[0] - (centers[1] - centers[0]) / 2]
            + [(left + right) / 2 for left, right in zip(centers, centers[1:])]
            + [centers[-1] + (centers[-1] - centers[-2]) / 2])


# Repeat each one-dimensional value across the rendered channel cross-section.
def _cell_color_row(values: list[float]) -> list[float]:
    return [value for cell in values for value in (cell, cell)]


# A single value colors the upstream or downstream end face.
def _end_color(value: float) -> list[list[float]]:
    return [[value, value], [value, value]]


# Build top, side, and end faces with the same property value through the full depth.
def _channel_surfaces(
    x_nodes: list[float], edges: list[float], width: float, depth: float,
    color_row: list[float], *, property_name: str, units: str | None,
    cmin: float, cmax: float,
) -> list[go.Surface]:
    """Six faces of one water volume, all at actual metre coordinates."""
    zeros = [0.0] * len(x_nodes)
    depths = [depth] * len(x_nodes)
    common = {
        "colorscale": COLORSCALE, "cmin": cmin, "cmax": cmax,
        "hovertemplate": ("Distance: %{x:.3g} m<br>Across river: %{y:.3g} m"
                          "<br>Height: %{z:.3g} m"
                          f"<br>{property_name}: " + "%{surfacecolor:.6g}<extra></extra>"),
    }
    return [
        go.Surface(
            x=[x_nodes, x_nodes], y=[zeros, [width] * len(x_nodes)],
            z=[depths, depths], surfacecolor=[color_row, color_row],
            colorbar={"title": f"{property_name} ({units})" if units else property_name,
                      "len": 0.62},
            name="Water surface", showscale=True, **common,
        ),
        go.Surface(
            x=[x_nodes, x_nodes], y=[zeros, zeros], z=[zeros, depths],
            surfacecolor=[color_row, color_row],
            name="Near bank", showscale=False, **common,
        ),
        go.Surface(
            x=[x_nodes, x_nodes],
            y=[[width] * len(x_nodes), [width] * len(x_nodes)],
            z=[zeros, depths], surfacecolor=[color_row, color_row],
            name="Far bank", showscale=False, **common,
        ),
        go.Surface(
            x=[[edges[0]] * 2, [edges[0]] * 2],
            y=[[0.0, width], [0.0, width]], z=[[0.0, 0.0], [depth, depth]],
            surfacecolor=_end_color(color_row[0]),
            name="Upstream end", showscale=False, **common,
        ),
        go.Surface(
            x=[[edges[-1]] * 2, [edges[-1]] * 2],
            y=[[0.0, width], [0.0, width]], z=[[0.0, 0.0], [depth, depth]],
            surfacecolor=_end_color(color_row[-1]),
            name="Downstream end", showscale=False, **common,
        ),
        go.Surface(
            x=[x_nodes, x_nodes], y=[zeros, [width] * len(x_nodes)],
            z=[zeros, zeros], surfacecolor=[color_row, color_row],
            name="River bed", showscale=False, **common,
        ),
    ]


# Use one fixed color scale for every time frame so an animation's colors remain comparable.
def build_river_3d_figure(
    result: SimulationResult,
    property_name: str,
    *,
    width_m: float,
    depth_m: float,
    length_m: float,
    units: str | None = None,
) -> go.Figure:
    """Build an animated 3D river with a fixed color scale across all times.

    The frame count equals the number of *printed* Excel output times. This is
    the temporal resolution available to all result visualizations.
    """
    if property_name not in result.values:
        raise KeyError(f"No results for {property_name}")
    values = result.values[property_name]
    centers = result.coordinates_m
    if not values or len(values) != len(result.time_seconds):
        raise ValueError("3D river requires one property row per printed time")
    if any(len(row) != len(centers) for row in values):
        raise ValueError("3D river property rows must match the cell count")
    edges = _cell_edges(centers, length_m)
    x_nodes = [x for left, right in zip(edges, edges[1:]) for x in (left, right)]
    all_values = [float(value) for row in values for value in row]
    cmin = min(all_values)
    cmax = max(all_values)
    if cmin == cmax:
        padding = max(1.0, abs(cmin) * 0.01)
        cmin -= padding
        cmax += padding

    color_rows = [_cell_color_row(row) for row in values]
    surfaces = _channel_surfaces(
        x_nodes, edges, width_m, depth_m, color_rows[0],
        property_name=property_name, units=units, cmin=cmin, cmax=cmax,
    )
    frames = [
        go.Frame(
            name=str(index),
            data=[
                go.Surface(surfacecolor=[colors, colors]),
                go.Surface(surfacecolor=[colors, colors]),
                go.Surface(surfacecolor=[colors, colors]),
                go.Surface(surfacecolor=_end_color(colors[0])),
                go.Surface(surfacecolor=_end_color(colors[-1])),
                go.Surface(surfacecolor=[colors, colors]),
            ],
            traces=[0, 1, 2, 3, 4, 5],
            layout=go.Layout(title=(f"{property_name} along the river · "
                                    f"{seconds:g} s ({seconds / 86400:.4g} d)")),
        )
        for index, (colors, seconds) in enumerate(zip(color_rows, result.time_seconds))
    ]
    duration_ms = max(90, min(600, round(8000 / max(1, len(frames) - 1))))
    steps: list[dict[str, Any]] = []
    for index, seconds in enumerate(result.time_seconds):
        label = f"{seconds / 86400:.4g} d"
        steps.append({
            "label": label,
            "method": "animate",
            "args": [[str(index)], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}],
        })

    fig = go.Figure(data=surfaces, frames=frames)
    # Preserve the entered width:depth proportion where legible while shortening
    # the much longer river in the display. Numeric coordinates are unchanged.
    cross_section = max(width_m, depth_m)
    shown_width = max(0.08, 0.8 * width_m / cross_section)
    shown_depth = max(0.08, 0.8 * depth_m / cross_section)
    fig.update_layout(
        title=f"{property_name} along the river · "
              f"{result.time_seconds[0]:g} s ({result.time_seconds[0] / 86400:.4g} d)",
        scene={
            "xaxis": {"title": "Distance downstream (m)", "range": [edges[0], edges[-1]]},
            "yaxis": {"title": "Across river (m)", "range": [0, width_m]},
            "zaxis": {"title": "Height (m)", "range": [0, depth_m]},
            "aspectmode": "manual",
            # Width and depth are exaggerated in the display so a long river
            # remains visible; axes retain physical metre values.
            "aspectratio": {"x": 3.8, "y": shown_width, "z": shown_depth},
            "camera": {"eye": {"x": 1.45, "y": 1.5, "z": 0.75}},
        },
        height=650,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": -0.07,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"fromcurrent": True, "mode": "immediate",
                                  "frame": {"duration": duration_ms, "redraw": True},
                                  "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": False},
                                     "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.16, "y": -0.05, "len": 0.82,
            "currentvalue": {"prefix": "Time: ", "visible": True},
            "steps": steps,
        }],
    )
    return fig
