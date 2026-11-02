"""Record disposable, freshly calculated Excel/VBA reference results.

Examples (Windows with Excel and pywin32 installed)::

    python tests/record_excel_oracle.py --list
    python tests/record_excel_oracle.py --case baseline_20
    python tests/record_excel_oracle.py --case advection_upwind
    python tests/record_excel_oracle.py --all

Every case starts with a byte-for-byte copy of the supplied XLSM. The original
workbook is never opened for writing. A separate Excel process runs Channel on
the copy; results are read directly from that live process, then it is closed
without saving. The parent process verifies that the source hash did not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Channel_TempOD&BOD&CO2AdvectionSourceCode2025_V3.xlsm"
FIXTURES = ROOT / "tests" / "fixtures" / "excel_oracle"
PROPERTIES = ("Temperature", "Generic", "DO", "BOD", "CO2")
INTEGRATED = ("IntegratedAeroResp(kg)", "IntegratedAnaeroResp(kg)")


def case_matrix() -> dict[str, dict[str, dict[str, Any]]]:
    """Small-cell cases; each case is an independent original-workbook copy.

    Null values mean clear the cell. Graph cells are reduced to valid positions
    for the 20-cell grid. Every case includes the baseline overrides.
    """
    base = {
        "Main": {"B3": 0.005, "B4": 60, "B5": 120, "B6": "semi", "B7": "no", "B10": "yes"},
        "River": {"B3": 20},
        "Graph": {"C2": 1, "D2": 5, "E2": 10, "F2": None, "G2": None},
        "Temperature": {"B20": 5, "C20": 10},
    }

    variants: dict[str, dict[str, dict[str, Any]]] = {
        "baseline_20": {},
        "exp_diffusion": {"Main": {"B6": "exp"}},
        "imp_diffusion": {"Main": {"B6": "imp"}},
        "no_diffusion": {"Main": {"B10": "no"}},
        "advection_upwind": {"Main": {"B7": "yes", "B8": "upwind"}},
        "advection_central": {"Main": {"B7": "yes", "B8": "central"}},
        "advection_quick": {"Main": {"B7": "yes", "B8": "QUICK"}},
        "advection_quick_up": {"Main": {"B7": "yes", "B8": "QUICK_UP", "B9": 4}},
        "river_geometry": {"River": {"B2": 8000, "B4": 6, "B5": 0.8, "B8": 0.12, "B10": 2.5}},
        "river_standard_diffusivity": {"River": {"B10": "standard"}},
        "duration_and_print": {"Main": {"B3": 0.01, "B4": 30, "B5": 90}},
        "temperature_default": {"Temperature": {"B17": "DEFAULT", "B16": 17}},
        "temperature_cell": {"Temperature": {"B17": "CELL", "B18": 7, "B19": 31}},
        "temperature_interval": {"Temperature": {"B20": 3, "C20": 15, "B21": 30}},
        "temperature_surface_off": {"Temperature": {"B7": "no"}},
        "temperature_sensible_off": {"Temperature": {"B8": "no"}},
        "temperature_latent_off": {"Temperature": {"B9": "no"}},
        "temperature_radiative_off": {"Temperature": {"B10": "no"}},
        "temperature_boundary": {"Temperature": {"B12": "no", "B13": 24, "B14": 10}},
        "atmosphere": {"Atmosphere": {"B3": 24, "B4": 2, "B6": 60, "B7": 1200, "B10": 25}},
        "discharge": {"Discharges": {"B3": 10, "B5": 0.1, "B6": 45, "B10": 10000}},
        "generic_cell": {"Generic": {"B13": "CELL", "B14": 4, "B15": 10000, "C14": None, "C15": None}},
        "generic_interval": {"Generic": {"B13": "INTERVAL", "B16": 5, "C16": 10, "B17": 5000, "D17": None}},
        "generic_decay": {"Generic": {"B8": 5}},
        "do_active": {"DO": {"B3": "Yes", "B8": "yes"}},
        "do_cell": {"DO": {"B3": "Yes", "B12": "CELL", "B13": 8, "B14": 15}},
        "do_interval": {"DO": {"B3": "Yes", "B12": "INTERVAL", "B15": 5, "C15": 10, "D15": None, "E15": None, "B16": 12, "D16": None}},
        "bod_active": {"DO": {"B3": "Yes"}, "BOD": {"B3": "Yes"}},
        "bod_anaerobic": {"DO": {"B3": "Yes"}, "BOD": {"B3": "Yes", "B11": "Yes"}},
        "co2_active": {"DO": {"B3": "Yes"}, "BOD": {"B3": "Yes"}, "CO2": {"B3": "Yes", "B8": "yes"}},
        "all_active_mixed": {
            "Main": {"B7": "yes", "B8": "upwind"},
            "DO": {"B3": "Yes", "B8": "yes"},
            "BOD": {"B3": "Yes", "B11": "Yes"},
            "CO2": {"B3": "Yes", "B8": "yes"},
            "Discharges": {"B5": 0.1, "B6": 40, "B7": 4, "B8": 50, "B9": 3, "B10": 1000},
        },
    }
    # Stress controls are deliberately isolated by default. Coupled cases
    # activate only the dependencies required by the particular variable.
    for time_method in ("exp", "imp", "semi"):
        for advection_type in ("upwind", "central", "QUICK", "QUICK_UP"):
            variants[f"stress_{time_method}_{advection_type.lower()}"] = {
                "Main": {"B6": time_method, "B7": "yes", "B8": advection_type,
                         "B9": 4, "B10": "yes"},
            }
    daylight = {"B3": 0.5, "B4": 1800, "B5": 7200}
    do_on = {"B3": "Yes", "B8": "yes"}
    bod_on = {"B3": "Yes"}
    co2_on = {"B3": "Yes", "B8": "yes"}
    variants.update({
        "stress_velocity_positive": {"Main": {"B7": "yes"}, "River": {"B9": 0.04}},
        "stress_velocity_negative": {"Main": {"B7": "yes"}, "River": {"B9": -0.02}},
        "stress_velocity_zero": {"Main": {"B7": "yes"}, "River": {"B9": 0}},
        "stress_advection_only": {"Main": {"B7": "yes", "B10": "no"}},
        "stress_neither_transport": {"Main": {"B7": "no", "B10": "no"}},
        "stress_atmos_air_temperature": {"Atmosphere": {"B3": 28}},
        "stress_atmos_humidity": {"Atmosphere": {"B6": 40}},
        "stress_atmos_solar": {"Main": daylight, "Atmosphere": {"B7": 1700}},
        "stress_atmos_sun_hours": {"Main": daylight, "Atmosphere": {"B11": 5, "B12": 20}},
        "stress_atmos_latitude": {"Main": daylight, "Atmosphere": {"B8": 45}},
        "stress_atmos_sky_cloud": {"Atmosphere": {"B9": -20, "B10": 50}},
        "stress_atmos_wind": {"Atmosphere": {"B4": 3}},
        "stress_o2_henry": {"DO": do_on, "Atmosphere": {"C23": 0.0017}},
        "stress_o2_pressure": {"DO": do_on, "Atmosphere": {"B15": 0.18}},
        "stress_co2_henry": {"DO": do_on, "BOD": bod_on,
                              "CO2": co2_on, "Atmosphere": {"D23": 0.05}},
        "stress_co2_pressure": {"DO": do_on, "BOD": bod_on,
                                 "CO2": co2_on, "Atmosphere": {"B16": 0.002}},
        "stress_generic_half_life": {"Generic": {"B8": 1.5, "C8": "HalfLife(Days)"}},
        "stress_generic_duplicate": {"Generic": {"B8": 2, "C8": "TDuplicate(Days)"}},
        "stress_generic_day_decay": {"Generic": {"B8": -0.5, "C8": "Day-1"}},
        "stress_generic_rt": {"Generic": {"B8": 2, "C8": "RT"}},
        "stress_generic_growth": {"Generic": {"B8": 1.2, "C8": "Day-1"}},
        "stress_bod_kinetics": {"DO": do_on, "BOD": {"B3": "Yes", "B7": 0.3,
                                                         "B8": 20, "B9": 1, "B10": 0.05}},
        "stress_bod_initial": {"DO": do_on, "BOD": {"B3": "Yes", "B15": "INTERVAL",
                                                       "B18": 5, "C18": 12,
                                                       "D18": None, "E18": None,
                                                       "B19": 100, "D19": None}},
        "stress_do_gas_off": {"DO": {"B3": "Yes", "B8": "no"}},
        "stress_do_initial_default": {"DO": {"B3": "Yes", "B8": "yes", "B11": 12}},
        "stress_co2_gas_off": {"DO": do_on, "BOD": bod_on,
                               "CO2": {"B3": "Yes", "B8": "no"}},
        "stress_co2_initial_cell": {"DO": do_on, "BOD": bod_on,
                                    "CO2": {"B3": "Yes", "B12": "CELL", "B13": 6, "B14": 4}},
        "stress_co2_initial_interval": {"DO": do_on, "BOD": bod_on,
                                        "CO2": {"B3": "Yes", "B12": "INTERVAL",
                                                "B15": 5, "C15": 10,
                                                "D15": None, "E15": None,
                                                "B16": 2, "D16": None}},
        "stress_discharge_temperature": {"Discharges": {"B5": 0.1, "B6": 45}},
        "stress_discharge_generic": {"Discharges": {"B5": 0.1, "B10": 15000}},
        "stress_discharge_do": {"DO": do_on,
                                "Discharges": {"B5": 0.1, "B7": 4}},
        "stress_discharge_bod": {"DO": do_on, "BOD": bod_on,
                                 "Discharges": {"B5": 0.1, "B8": 100}},
        "stress_discharge_co2": {"DO": do_on, "BOD": bod_on, "CO2": co2_on,
                                 "Discharges": {"B5": 0.1, "B9": 4}},
        "stress_discharge_flow": {"Discharges": {"B5": 0.2, "B6": 40, "B10": 10000}},
        "stress_discharge_cell": {"Discharges": {"B3": 15, "B5": 0.1,
                                                  "B6": 40, "B10": 10000}},
        "stress_temperature_m_interval": {
            "Temperature": {"A20": "(X1,X2)_intervalValues(m):",
                            "B17": "INTERVAL", "B20": 1250.4,
                            "C20": 3750.6, "B21": 9},
        },
        "stress_henry_short_columns": {
            "DO": do_on, "BOD": bod_on, "CO2": co2_on,
            "Atmosphere": {"C24": None, "D24": None},
        },
        "stress_graph_dense": {"Graph": {"C2": 1, "D2": 2, "E2": 3, "F2": 10,
                                         "C3": 0, "D3": 25, "E3": 50,
                                         "F3": 75, "G3": 100, "H3": None}},
        # Coupled probes deliberately move several independent controls at
        # once. They use a short grid so every printed cell remains cheap to
        # compare, but include high transport numbers, long daylight exposure,
        # large source terms, and all five interacting properties.
        "combo_hot_shallow_daylight": {
            "Main": {"B3": 0.5, "B4": 1800, "B5": 3600, "B6": "semi",
                     "B7": "yes", "B8": "QUICK_UP", "B9": 3, "B10": "yes"},
            "River": {"B4": 2, "B5": 0.12, "B8": 0.7, "B9": 0.15, "B10": 1},
            "Atmosphere": {"B3": 38, "B4": 7, "B6": 25, "B7": 1650,
                           "B8": 12, "B9": 30, "B10": 70, "B11": 5, "B12": 19},
            "Generic": {"B8": 0.7, "C8": "HalfLife(Days)"},
            "DO": {"B3": "Yes", "B8": "yes"},
            "BOD": {"B3": "Yes", "B7": 0.35, "B9": 2},
            "CO2": {"B3": "Yes", "B8": "yes"},
            "Discharges": {"B3": 12, "B5": 0.25, "B6": 48,
                           "B7": 1, "B8": 350, "B9": 8, "B10": 50000},
        },
        "combo_cold_deep_reverse": {
            "Main": {"B3": 0.03, "B4": 120, "B5": 240, "B6": "imp",
                     "B7": "yes", "B8": "central", "B10": "yes"},
            "River": {"B2": 5000, "B3": 25, "B4": 15, "B5": 4,
                      "B8": 0.4, "B9": -0.15, "B10": 30},
            "Atmosphere": {"B3": -5, "B4": 4, "B6": 90, "B9": -60,
                           "B10": 85, "B11": 7, "B12": 17},
            "Temperature": {"B17": "DEFAULT", "B16": 4},
            "Generic": {"B8": 1.3, "C8": "TDuplicate(Days)"},
            "DO": {"B3": "Yes", "B8": "yes", "B11": 3},
            "BOD": {"B3": "Yes", "B11": "Yes", "B7": 0.22, "B8": 8},
            "CO2": {"B3": "Yes", "B8": "yes", "B12": "CELL",
                    "B13": 7, "B14": 3},
            "Discharges": {"B3": 6, "B5": 0.35, "B6": 12,
                           "B7": 2, "B8": 150, "B9": 5, "B10": 5000},
        },
        "combo_high_diffusion_implicit": {
            "Main": {"B3": 0.05, "B4": 300, "B5": 600, "B6": "imp",
                     "B7": "yes", "B8": "upwind", "B10": "yes"},
            "River": {"B2": 1000, "B8": 0.1, "B9": 0.03, "B10": 150},
            "Atmosphere": {"B3": 25, "B4": 3, "B6": 40, "B10": 50},
            "Generic": {"B8": 0.9, "C8": "Day-1", "B13": "CELL",
                        "B14": 3, "B15": 100000, "C14": None, "C15": None},
            "DO": {"B3": "Yes", "B8": "yes"},
            "BOD": {"B3": "Yes", "B7": 0.4, "B9": 2},
            "CO2": {"B3": "Yes", "B8": "yes"},
            "Discharges": {"B3": 16, "B5": 0.2, "B6": 36,
                           "B7": 0.5, "B8": 200, "B9": 7, "B10": 10000},
        },
        "combo_fast_reverse_quick": {
            "Main": {"B3": 0.025, "B4": 45, "B5": 135, "B6": "semi",
                     "B7": "yes", "B8": "QUICK", "B9": 4, "B10": "yes"},
            "River": {"B2": 1500, "B3": 30, "B4": 4, "B5": 0.3,
                      "B8": 0.3, "B9": -0.65, "B10": 3},
            "Generic": {"B8": -0.8, "C8": "Day-1", "B13": "INTERVAL",
                        "B16": 3, "C16": 18, "B17": 10000, "D17": None},
            "Temperature": {"B12": "no", "B13": 33, "B14": 4},
            "DO": {"B3": "Yes", "B8": "no"},
            "BOD": {"B3": "Yes", "B11": "Yes"},
            "CO2": {"B3": "Yes", "B8": "no"},
            "Discharges": {"B3": 22, "B5": 0.12, "B6": 42,
                           "B7": 1, "B8": 70, "B9": 4, "B10": 20000},
        },
        "combo_irregular_explicit": {
            "Main": {"B3": 0.017, "B4": 17, "B5": 43, "B6": "exp",
                     "B7": "yes", "B8": "central", "B10": "yes"},
            "River": {"B2": 900, "B3": 30, "B4": 3, "B5": 0.25,
                      "B8": 0.2, "B9": 0.4, "B10": 5},
            "Atmosphere": {"B3": 27, "B4": 5, "B6": 45, "B9": -10},
            "Generic": {"B8": 2, "C8": "RT"},
            "DO": {"B3": "Yes", "B8": "yes"},
            "BOD": {"B3": "Yes", "B7": 0.3, "B9": 1},
            "CO2": {"B3": "Yes", "B8": "yes"},
            "Discharges": {"B3": 11, "B5": 0.1, "B6": 41,
                           "B7": 1.5, "B8": 100, "B9": 6, "B10": 9000},
        },
        "combo_high_bod_load": {
            "Main": {"B3": 0.1, "B4": 60, "B5": 600, "B6": "semi",
                     "B7": "yes", "B8": "upwind", "B10": "yes"},
            "River": {"B2": 4000, "B3": 40, "B4": 4, "B5": 0.4,
                      "B8": 1, "B9": 0.15, "B10": 30},
            "Atmosphere": {"B3": 31, "B4": 6, "B6": 35,
                           "B15": 0.18, "B16": 0.002},
            "Generic": {"B8": 0.4, "C8": "Day-1"},
            "DO": {"B3": "Yes", "B8": "yes", "B11": 0.5},
            "BOD": {"B3": "Yes", "B11": "Yes", "B7": 0.6,
                    "B8": 15, "B9": 3, "B10": 0.03, "B14": 800},
            "CO2": {"B3": "Yes", "B8": "yes", "B11": 15},
            "Discharges": {"B3": 25, "B5": 1, "B6": 40,
                           "B7": 0, "B8": 1200, "B9": 20, "B10": 50000},
        },
        # These last two cases probe the spreadsheet's gap handling. A blank
        # time-series cell can trigger its validation dialog; run explicitly.
        "stress_graph_time_gap": {"Graph": {"C2": 1, "D2": None, "E2": 10}},
        "stress_graph_profile_gap": {"Graph": {"C3": 0, "D3": None, "E3": 100,
                                               "F3": None, "G3": None, "H3": None}},
    })
    matrix = {}
    for name, delta in variants.items():
        merged = {sheet: dict(cells) for sheet, cells in base.items()}
        for sheet, cells in delta.items():
            merged.setdefault(sheet, {}).update(cells)
        matrix[name] = merged
    return matrix


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def simple(value: Any) -> Any:
    """Normalize pywin32 COM variants to JSON-compatible scalars/lists."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [simple(x) for x in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def matrix(value: Any) -> list[list[Any]]:
    value = simple(value)
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return value
        return [value]
    return [[value]]


def read_sheet_rows(sheet: Any, nc: int) -> dict[str, Any]:
    # The macro clears rows 22 onward of active sheets before writing. Row 22
    # contains one header for each cell, so NC+3 is the exact output width.
    # A time value is written in column A for every output row; xlUp avoids
    # Excel Find's application-global sticky settings and search origin.
    last_row = int(sheet.Cells(sheet.Rows.Count, 1).End(-4162).Row)
    if last_row < 25:
        raise RuntimeError(f"{sheet.Name}: macro did not write an output header")
    width = nc + 3
    values = matrix(sheet.Range(sheet.Cells(22, 1), sheet.Cells(last_row, width)).Value2)
    if len(values) != last_row - 21 or any(len(row) != width for row in values):
        raise RuntimeError(f"{sheet.Name}: malformed output matrix")
    if values[0][0] != "Cell Number":
        raise RuntimeError(f"{sheet.Name}: missing fresh output header: {values[0][0]!r}")
    times = [row[0] for row in values[4:]]
    if not times or times[0] != 0:
        raise RuntimeError(f"{sheet.Name}: missing time-zero output")
    if any(not isinstance(t, (int, float)) for t in times):
        raise RuntimeError(f"{sheet.Name}: nonnumeric time row")
    return {
        "address": f"A22:{sheet.Cells(last_row, width).Address.replace('$', '')}",
        "width": width,
        "height": len(values),
        "headers": {"cell_numbers": values[0], "coordinates": values[1], "labels": values[3]},
        "data": values[4:],
    }


def read_charts(sheet: Any) -> list[dict[str, Any]]:
    charts = []
    objects = sheet.ChartObjects()
    for i in range(1, objects.Count + 1):
        obj = objects.Item(i)
        chart = obj.Chart
        title = chart.ChartTitle.Text if chart.HasTitle else None
        series = []
        collection = chart.SeriesCollection()
        for j in range(1, collection.Count + 1):
            item = collection.Item(j)
            series.append({
                "name": simple(item.Name),
                "formula": simple(item.Formula),
                "x_values": simple(item.XValues),
                "y_values": simple(item.Values),
            })
        charts.append({"object_name": obj.Name, "title": title,
                       "chart_type": int(chart.ChartType), "series": series})
    return charts


def window_text(hwnd: int, win32gui: Any) -> str:
    parts = []
    def collect(child: int, _: Any) -> None:
        label = win32gui.GetWindowText(child).strip()
        if label:
            parts.append(label)
    win32gui.EnumChildWindows(hwnd, collect, None)
    return " | ".join(parts)


def dialog_watcher(pid: int, stop: threading.Event, seen: list[dict[str, str]],
                   fatal: list[str]) -> None:
    import win32api
    import win32con
    import win32gui
    import win32process

    allowed = ("The Application Ended", "Diffusivity value to be used is ")
    encountered: set[tuple[int, str]] = set()
    while not stop.wait(0.15):
        windows: list[int] = []
        win32gui.EnumWindows(lambda hwnd, acc: acc.append(hwnd), windows)
        for hwnd in windows:
            if not win32gui.IsWindowVisible(hwnd):
                continue
            _, owner = win32process.GetWindowThreadProcessId(hwnd)
            if owner != pid or win32gui.GetClassName(hwnd) != "#32770":
                continue
            message = window_text(hwnd, win32gui)
            key = (hwnd, message)
            if key not in encountered:
                encountered.add(key)
                title = win32gui.GetWindowText(hwnd)
                seen.append({"title": title, "text": message})
            if any(expected in message for expected in allowed):
                # Only the known normal completion and standard diffusivity
                # information dialogs are acknowledged automatically.
                buttons: list[int] = []
                win32gui.EnumChildWindows(hwnd, lambda child, acc: acc.append(child), buttons)
                ok = next((child for child in buttons
                           if win32gui.GetClassName(child) == "Button"
                           and win32gui.GetWindowText(child).strip() == "OK"), None)
                if ok is None:
                    fatal.append(f"Known dialog without OK button: {message}")
                    return
                win32gui.PostMessage(ok, win32con.BM_CLICK, 0, 0)
            else:
                if key in encountered and seen and seen[-1]["text"] != message:
                    continue
                fatal.append(f"Unexpected Excel dialog: {title}: {message}")
                try:
                    handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, False, pid)
                    win32api.TerminateProcess(handle, 1)
                    win32api.CloseHandle(handle)
                except Exception as exc:
                    fatal.append(f"Could not stop isolated Excel process: {exc}")
                return


def run_worker(args: argparse.Namespace) -> None:
    import pythoncom
    import win32com.client
    import win32process

    pythoncom.CoInitialize()
    app = None
    book = None
    stop = threading.Event()
    watcher = None
    dialogs: list[dict[str, str]] = []
    fatal: list[str] = []
    try:
        app = win32com.client.DispatchEx("Excel.Application")
        _, pid = win32process.GetWindowThreadProcessId(app.Hwnd)
        Path(args.pid_file).write_text(str(pid), encoding="utf-8")
        app.Visible = False
        app.DisplayAlerts = False
        app.EnableEvents = False
        app.AskToUpdateLinks = False
        app.AutomationSecurity = 1  # msoAutomationSecurityLow, on disposable local copy
        watcher = threading.Thread(target=dialog_watcher,
                                   args=(pid, stop, dialogs, fatal), daemon=True)
        watcher.start()
        book = app.Workbooks.Open(str(args.copy), UpdateLinks=0, ReadOnly=False)
        for sheet_name, edits in case_matrix()[args.case].items():
            sheet = book.Worksheets(sheet_name)
            for address, value in edits.items():
                sheet.Range(address).Value2 = value
        app.CalculateFull()
        # Save only the disposable copy so openpyxl sees the edited input
        # cells and Excel's freshly recalculated formula caches. Capture this
        # before Channel populates results or changes calculation settings.
        book.Save()
        sys.path.insert(0, str(ROOT))
        from transportmodel.workbook import read_workbook
        config = read_workbook(args.copy)
        # The workbook button calls Run, whose dispatcher currently resolves
        # to this local Channel entry point. Explicit target avoids stale C2.
        app.Run(f"'{book.Name}'!Module1.Channel")
        if fatal:
            raise RuntimeError("; ".join(fatal))
        nc = int(book.Worksheets("River").Range("B3").Value2)
        active = [name for name in PROPERTIES
                  if str(book.Worksheets(name).Range("B3").Value2).strip().lower() == "yes"]
        if not active:
            raise RuntimeError("No active properties after macro run")
        outputs = {name: read_sheet_rows(book.Worksheets(name), nc) for name in active}
        if "BOD" in active:
            outputs.update({name: read_sheet_rows(book.Worksheets(name), nc) for name in INTEGRATED})
        result = {
            "case": args.case,
            "source_sha256": args.source_sha256,
            "excel_pid": pid,
            "workbook_name": book.Name,
            "edits": case_matrix()[args.case],
            "config": config,
            "effective_inputs": {
                "river_velocity": simple(book.Worksheets("River").Range("B9").Value2),
                "co2_partial_pressure": simple(book.Worksheets("Atmosphere").Range("B16").Value2),
            },
            "active_properties": active,
            "outputs": outputs,
            "river_diagnostics": matrix(book.Worksheets("River").Range("F2:H9").Value2),
            "graphs": read_charts(book.Worksheets("Graph")),
            "dialogs": dialogs,
        }
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                               allow_nan=False), encoding="utf-8")
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=2)
        try:
            if book is not None:
                book.Close(SaveChanges=False)
        finally:
            if app is not None:
                app.Quit()
        pythoncom.CoUninitialize()


def kill_isolated_excel(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text(encoding="utf-8"))
    # The PID was obtained from DispatchEx's own Excel window, not from the
    # user's existing Excel session. taskkill is only a last-resort watchdog.
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True)


def record_case(name: str, timeout: int) -> Path:
    original_hash = sha256(SOURCE)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    destination = FIXTURES / f"{name}.json"
    with tempfile.TemporaryDirectory(prefix=f"transport_oracle_{name}_") as work:
        directory = Path(work)
        copy = directory / SOURCE.name
        output = directory / "result.json"
        pid_file = directory / "excel.pid"
        shutil.copy2(SOURCE, copy)
        if sha256(copy) != original_hash:
            raise RuntimeError("Disposable copy hash differs from source")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--case", name, "--copy", str(copy), "--output", str(output),
                   "--pid-file", str(pid_file), "--source-sha256", original_hash]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_isolated_excel(pid_file)
            raise RuntimeError(f"Excel oracle case {name} exceeded {timeout}s") from exc
        finally:
            if sha256(SOURCE) != original_hash:
                raise RuntimeError("Original workbook hash changed during oracle recording")
        if run.returncode or not output.exists():
            kill_isolated_excel(pid_file)
            raise RuntimeError(f"Excel oracle case {name} failed:\n{run.stdout}\n{run.stderr}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["source_sha256"] != original_hash:
            raise RuntimeError("Oracle fixture source hash mismatch")
        shutil.copy2(output, destination)
        print(f"{name}: {len(result['outputs'])} sheets, "
              f"{sum(len(v['data']) for v in result['outputs'].values())} time rows, "
              f"{len(result['graphs'])} charts -> {destination}")
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show reproducible cases")
    parser.add_argument("--case", choices=case_matrix())
    parser.add_argument("--all", action="store_true", help="record all cases sequentially")
    parser.add_argument("--stress", action="store_true",
                        help="record second stress matrix, excluding graph-gap probes")
    parser.add_argument("--graph-gaps", action="store_true",
                        help="record gap probes, which may expose macro validation dialogs")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--missing", action="store_true",
                        help="with --all or --stress, only record fixtures not already present")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--copy", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pid-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    elif args.list:
        for case in case_matrix():
            print(case)
    elif args.all:
        for case in case_matrix():
            if not args.missing or not (FIXTURES / f"{case}.json").exists():
                record_case(case, args.timeout)
    elif args.stress:
        for case in case_matrix():
            if case.startswith("stress_") and "graph_" not in case or case == "stress_graph_dense":
                if not args.missing or not (FIXTURES / f"{case}.json").exists():
                    record_case(case, args.timeout)
    elif args.graph_gaps:
        for case in ("stress_graph_time_gap", "stress_graph_profile_gap"):
            record_case(case, args.timeout)
    else:
        record_case(args.case or "baseline_20", args.timeout)


if __name__ == "__main__":
    main()
