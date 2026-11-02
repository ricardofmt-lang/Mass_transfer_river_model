"""Build workbook-style charts from the Python river-model results.

``build_charts`` returns ordered, serializable chart specifications. Its
``series`` entries retain the macro's names, placement, axes, and formula
metadata. X/Y values always come from the current simulation, never from an
old worksheet. This corrects the macro's stale-CO2 chart selection bug.
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

from .numerical import SimulationResult


# Order of ActiveWorkbook.Worksheets in the supplied XLSM. The VBA filters this
# order separately for each graph group; it does not use calculation order.
WORKBOOK_SHEET_ORDER = (
    "Main", "River", "Atmosphere", "Discharges", "Temperature", "DO", "CO2",
    "BOD", "Generic", "Graph", "IntegratedAeroResp(kg)",
    "IntegratedAnaeroResp(kg)", "Sheet1", "Sheet2",
)
COLOR_INDEX = {
    "Temperature": 41, "Zoo": 3, "Detritus": 52, "Nutrient": 44,
    "Phyto": 50, "DO": 45, "CO2": 55, "BOD": 3, "Generic": 41,
}
# Colors read from the supplied workbook palette through Excel COM.
COLOR_RGB = {3: "#FF0000", 33: "#00CCFF", 41: "#3366FF",
             45: "#FF9900", 55: "#333399"}


# Charts should include only properties calculated in this simulation.
def _eligible_sheets(result: SimulationResult, order: tuple[str, ...]) -> list[str]:
    """Only chart properties actually calculated in this simulation."""
    return [name for name in order if name in result.values]


# Convert one-based worksheet columns to Excel A1 references.
def _column_letter(column: int) -> str:
    letters = ""
    while column:
        column, digit = divmod(column - 1, 26)
        letters = chr(65 + digit) + letters
    return letters


# Quote worksheet names so formula metadata remains valid when names contain spaces.
def _sheet_ref(name: str) -> str:
    # Excel quotes names that can be parsed as an A1 cell reference; CO2 is
    # one such name even though it otherwise looks like an identifier.
    bare = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))
    cell_like = bool(re.fullmatch(r"[A-Za-z]{1,3}[1-9][0-9]*", name))
    return name if bare and not cell_like else "'" + name.replace("'", "''") + "'"


# Build the exact worksheet range represented by a plotted series.
def _range(name: str, first_col: int, first_row: int, last_col: int, last_row: int) -> str:
    sheet = _sheet_ref(name)
    return (f"{sheet}!${_column_letter(first_col)}${first_row}:"
            f"${_column_letter(last_col)}${last_row}")


# Store X and Y values alongside series names and original Excel formula metadata.
def _series(name: str, x: list[float | None], y: list[float | None],
            x_range: str, y_range: str, index: int) -> dict[str, Any]:
    return {
        "name": name,
        "formula": f'=SERIES("{name.replace(chr(34), chr(34) * 2)}",{x_range},{y_range},{index})',
        "x_values": x,
        "y_values": y,
    }


# VBA rounds graph selector values before turning them into row or cell indices.
def _vba_number(value: float) -> str:
    """VBA's default string form after ``Round(value, 2)`` for chart labels."""
    # VBA's decimal midpoint rule differs from Python's binary-float round:
    # Round(0.025, 2) is 0.02 in the workbook, while round(0.025, 2) is 0.03.
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    result = format(rounded, "f").rstrip("0").rstrip(".")
    return "0" if result in ("", "-", "-0") else result


# The original macro accepts several numeric cell types and coerces them to Double.
def _vba_cell_number(value: Any) -> float:
    # A blank worksheet cell is numerically zero in these VBA assignments.
    return 0.0 if value is None or value == "" else float(value)


# A percentage selector chooses a printed time row, not a simulation step.
def _percentage_row(raw: Any, row_count: int) -> int:
    # CInt uses banker's rounding; assignment of the line to Long rounds again.
    percentage = int(round(_vba_cell_number(raw)))
    return int(round(26 + percentage / 100 * (row_count - 1))) - 26


# Ignore blank values when deriving chart axis limits.
def _axis_bounds(numbers: list[float | None]) -> tuple[float | None, float | None]:
    values = [float(value) for value in numbers if value is not None]
    if not values:
        return None, None
    ymin = min(values)
    ymax = max(values)
    if ymin > 0:
        ymin = 0.9 * round(ymin / 10, 2) * 10
    else:
        ymin = 1.1 * round(ymin / 10, 2) * 10
    if ymax > 0:
        ymax = 1.1 * round(ymax / 10, 2) * 10
    else:
        # The VBA uses the already adjusted ymin here, not ymax.
        ymax = 0.9 * round(ymin / 10, 2) * 10
    return ymin, ymax


# Bundle title, style, axis ranges, and series into a serializable chart specification.
def _chart(title: str, name: str, chart_type: int, series: list[dict[str, Any]],
           left: float, top: float, axis_values: list[float | None] | None,
           no_major_gridlines: bool, source_status: str) -> dict[str, Any]:
    ymin, ymax = _axis_bounds(axis_values) if axis_values is not None else (None, None)
    return {
        "object_name": "Chart 1",
        "title": title,
        "chart_type": chart_type,  # 74: xlXYScatterLines; 72: xlXYScatterSmooth
        "series": series,
        "x_axis_title": "Days" if "time" in title or title.startswith("Time Series") else "Distance to origin (m)",
        "y_axis_title": name,
        "y_axis_minimum": ymin,
        "y_axis_maximum": ymax,
        "y_axis_number_format": "#####.00" if axis_values is not None else None,
        "left": left,
        "top": top,
        "width": 500,
        "height": 325,
        "line_color_index": COLOR_INDEX.get(name, 33),
        "line_color_hex": COLOR_RGB.get(COLOR_INDEX.get(name, 33), "#00CCFF"),
        "formatted_series_index": len(series),
        "no_major_gridlines": no_major_gridlines,
        "source_status": source_status,
    }


# Follow the macro's chart order, but read data from this run's result arrays.
def build_charts(
    result: SimulationResult,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return workbook-style charts in VBA creation order for computed data."""
    graph = config.get("graph") or {}
    sheet_order = tuple(graph.get("sheet_order") or WORKBOOK_SHEET_ORDER)
    sheets = _eligible_sheets(result, sheet_order)
    no_major_gridlines = bool(graph.get("no_major_gridlines", False))
    nc = len(result.coordinates_m)
    row_count = len(result.time_seconds)
    last_row = 25 + row_count
    x_days = [time / 86400 for time in result.time_seconds]
    x_space: list[float | None] = [*result.coordinates_m, None]

    # Align sparse chart selectors with their expected workbook column count.
    def _pad(items: list[Any] | None, size: int) -> list[Any]:
        values = list(items or [])[:size]
        return values + [None] * (size - len(values))

    # Read all printed rows for one property in workbook order.
    def rows(name: str) -> list[list[float | None]]:
        data = result.values[name]
        return [_pad(list(row), nc + 1)
                for row in list(data[:row_count]) + [[None] * (nc + 1)] * max(0, row_count - len(data))]

    # Convert printed seconds to days for the workbook's time axis.
    def sheet_days(name: str) -> list[float | None]:
        return x_days.copy()

    # Return physical cell centres for the workbook's distance axis.
    def sheet_space(name: str) -> list[float | None]:
        return x_space.copy()

    raw_cells = list(graph.get("time_series_cells") or [])
    cells: list[int] = []
    for raw in raw_cells:
        value = _vba_cell_number(raw)
        if value < 0 or value > nc:
            break  # VBA truncates the list at the first invalid cell.
        cells.append(int(round(value)))
    raw_percentages = list(graph.get("longitudinal_profiles_percent") or [])
    for raw in raw_percentages:
        value = _vba_cell_number(raw)
        if value < 0 or value > 100:
            raise ValueError(f"Time Percentage for graph has to be between 0 and 100: {raw}")
    profile_rows = [_percentage_row(raw, row_count) for raw in raw_percentages]

    charts: list[dict[str, Any]] = []
    left = 0.0
    top = 0.0

    if cells:
        left = 100.0
        top = 100.0
        for name in sheets:
            data = rows(name)
            days = sheet_days(name)
            x_range = _range(name, 2, 26, 2, last_row)
            series = []
            for index, cell in enumerate(cells, start=1):
                y = [row[cell - 1] if cell else None for row in data]
                y_range = _range(name, 3 + cell, 26, 3 + cell, last_row)
                series.append(_series(f" @Cell {cell}", days.copy(), y, x_range, y_range, index))
            lower, upper = sorted((cells[0], cells[-1]))
            axis_values = [row[cell - 1] for row in data for cell in range(max(1, lower), upper + 1)]
            charts.append(_chart(f"{name} vs time in cell ", name, 74, series,
                                 left, top, axis_values, no_major_gridlines, "computed"))
            left += 550
        left = 100.0
        top += 350

    if profile_rows:
        for name in sheets:
            data = rows(name)
            days = sheet_days(name)
            space = sheet_space(name)
            x_range = _range(name, 4, 23, 4 + nc, 23)
            series = []
            for index, row_index in enumerate(profile_rows, start=1):
                y: list[float | None] = data[row_index].copy()
                y_range = _range(name, 4, 26 + row_index, 4 + nc, 26 + row_index)
                label = f" @Time {_vba_number(days[row_index] or 0.0)} Days"
                series.append(_series(label, space.copy(), y, x_range, y_range, index))
            first, last = sorted((profile_rows[0], profile_rows[-1]))
            axis_values = [value for row in data[first:last + 1] for value in row]
            charts.append(_chart(f"{name} vs Distance ", name, 74, series,
                                 left, top, axis_values, no_major_gridlines, "computed"))
            left += 550
        left = 100.0
        top += 350

    for cell in cells:
        for name in sheets:
            data = rows(name)
            days = sheet_days(name)
            y = [row[cell - 1] if cell else None for row in data]
            x_range = _range(name, 2, 26, 2, last_row)
            y_range = _range(name, 3 + cell, 26, 3 + cell, last_row)
            series = [_series(name, days, y, x_range, y_range, 1)]
            charts.append(_chart(f"Time Series at Cell {cell}", name, 72, series,
                                 left, top, None, no_major_gridlines, "computed"))
            left += 550
        top += 350
        left = 100.0

    for row_index in profile_rows:
        for name in sheets:
            data = rows(name)
            days = sheet_days(name)
            space = sheet_space(name)
            y = data[row_index].copy()
            x_range = _range(name, 4, 23, 4 + nc, 23)
            y_range = _range(name, 4, 26 + row_index, 4 + nc, 26 + row_index)
            series = [_series(name, space, y, x_range, y_range, 1)]
            title = f"Longitudinal Profile {_vba_number(days[row_index] or 0.0)} days after start"
            charts.append(_chart(title, name, 72, series,
                                 left, top, None, no_major_gridlines, "computed"))
            left += 550
        top += 350
        left = 100.0

    return charts
