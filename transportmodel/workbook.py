"""Read the river model's Excel input cells into an editable JSON configuration.

The VBA searches exact labels in column A.  In particular, ``PropertyName: ``
in the supplied workbook has a trailing space and is *not* read by its macro.
The configuration keeps the displayed cell and the label match separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook


PROPERTIES = ("Generic", "Temperature", "DO", "BOD", "CO2")
DISCHARGE_LABELS = {
    "Temperature": "DischargeTemperatures:",
    "DO": "DischargeConcentrations_DO:",
    "BOD": "DischargeConcentrations_BOD:",
    "CO2": "DischargeConcentrations_CO2:",
    "Generic": "DischargeGeneric:",
}


def _labels(sheet: Any, limit: int = 50) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in range(1, limit + 1):
        value = sheet.cell(row, 1).value
        if isinstance(value, str):
            labels.setdefault(value, row)
    return labels


def _cell(sheet: Any, labels: dict[str, int], label: str, column: int = 2,
          default: Any = None) -> Any:
    row = labels.get(label)
    return sheet.cell(row, column).value if row is not None else default


def _column_values(sheet: Any, row: int | None, start: int = 2) -> list[Any]:
    if row is None:
        return []
    values: list[Any] = []
    for col in range(start, min(sheet.max_column, 16384) + 1):
        value = sheet.cell(row, col).value
        if value is None:
            break
        values.append(value)
    return values


def _graph_values(sheet: Any, row: int | None) -> list[Any]:
    """Excel counts to the rightmost nonblank, retaining gaps as zero inputs."""
    if row is None:
        return []
    last = max((col for col in range(3, sheet.max_column + 1)
                if sheet.cell(row, col).value is not None), default=2)
    return [sheet.cell(row, col).value for col in range(3, last + 1)]


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _yes(value: Any, *, short: bool = False) -> bool:
    return value in (("YES", "Yes", "yes", "Y", "y") if short else ("YES", "Yes", "yes"))


def _read_property(sheet: Any, cells: int) -> dict[str, Any]:
    labels = _labels(sheet)
    get = lambda label, default=None: _cell(sheet, labels, label, default=default)
    init_type = str(get("InitType:", "DEFAULT") or "DEFAULT").upper()
    cell_row = labels.get("CellNumber:")
    cell_values = []
    if cell_row is not None:
        for col in range(2, min(sheet.max_column, 16384) + 1):
            index = sheet.cell(cell_row, col).value
            if index is None:
                break
            cell_values.append({"cell": index, "value": sheet.cell(cell_row + 1, col).value})
    interval_label = min((x for x in ("(X1,X2)_intervalValues(m):",
                                    "(X1,X2)_intervalValues(CELL):") if x in labels),
                         key=labels.get, default=None)
    intervals = []
    if interval_label:
        row = labels[interval_label]
        for col in range(2, min(sheet.max_column, 16384), 2):
            start = sheet.cell(row, col).value
            end = sheet.cell(row, col + 1).value
            value = sheet.cell(row + 1, col).value
            if start is None or end is None or value is None:
                break
            intervals.append({"start": start, "end": end, "value": value})
    return {
        "name_cell": sheet.cell(1, 2).value,
        "name_label_matches_vba": "PropertyName:" in labels,
        "units": get("PropertyUnits:", ""),
        "active": _yes(get("PropertyActive:"), short=True),
        "active_label_present": "PropertyActive:" in labels,
        "default": get("DefaultValue:", 0),
        "maximum": get("MaximumValue:", 0),
        "maximum_label_present": "MaximumValue:" in labels,
        "minimum": get("MinimumValue:", 0),
        "minimum_label_present": "MinimumValue:" in labels,
        "cyclic": _yes(get("CyclicBoundary:"), short=True),
        "cyclic_label_present": "CyclicBoundary:" in labels,
        "left_boundary": get("LeftBoundaryValue:", 0),
        "right_boundary": get("RightBoundaryValue:", 0),
        "free_surface_flux": _yes(get("FreeSurfaceFlux:")),
        "sensible_heat": not ("FreeSurfaceSensibeHeatFlux:" in labels and
                              not _yes(get("FreeSurfaceSensibeHeatFlux:"))),
        "latent_heat": not ("FreeSurfaceLatentHeatFlux:" in labels and
                            not _yes(get("FreeSurfaceLatentHeatFlux:"))),
        "radiative_heat": not ("FreeSurfaceRadiativeHeatFlux:" in labels and
                               not _yes(get("FreeSurfaceRadiativeHeatFlux:"))),
        "anaerobic_respiration": _yes(get("AnaerobicRespiration:"), short=True),
        "bod_oxygen_half_saturation": get("O2SemiSaturationConc:", 0),
        "bod_logistic_max": get("MaxValLogisticFormulation:", 10000000),
        "bod_logistic_max_label_present": "MaxValLogisticFormulation:" in labels,
        "bod_growth_per_day": get("GrowthRateLogisticFormulation:", 0),
        "bod_decay_per_day": get("Decay_Rate:", 0),
        "bod_decay_label_present": "Decay_Rate:" in labels,
        "generic_rate": get("Decay/Growth_Rate:", 0),
        "generic_rate_label_present": "Decay/Growth_Rate:" in labels,
        "generic_rate_units": _cell(sheet, labels, "Decay/Growth_Rate:", 3, "T90(Hours)"),
        "initial": {
            "type": init_type,
            "default_cell_value": get("DefaultCellValue:", None),
            "default_cell_label_present": "DefaultCellValue:" in labels,
            "cells": cell_values,
            "interval_kind": "m" if interval_label and interval_label.endswith("(m):") else "CELL",
            "intervals": intervals,
        },
    }


def read_workbook(path: str | Path) -> dict[str, Any]:
    """Extract every numerical input used by ``Channel`` from an XLSM copy.

    The two formula cells used as inputs are evaluated from their literal
    formulas, avoiding stale openpyxl cached values after an input edit.
    Other formulas can be read from Excel's saved calculation cache.
    """
    path = Path(path)
    # A normal worksheet makes the VBA-like repeated label searches cheap.
    # Read-only openpyxl reparses the sheet XML for every random ``cell`` read.
    wb = load_workbook(path, read_only=False, data_only=False, keep_vba=False)
    cache = None
    try:
        sheets = {name: wb[name] for name in ("Main", "River", "Atmosphere", "Discharges", "Graph", *PROPERTIES)}
        labels = {name: _labels(sheet, 20 if name == "Main" else 50) for name, sheet in sheets.items()}
        def val(name: str, label: str, col: int = 2, default: Any = None) -> Any:
            return _cell(sheets[name], labels[name], label, col, default)
        def cached(name: str, label: str) -> Any:
            nonlocal cache
            if cache is None:
                cache = load_workbook(path, read_only=False, data_only=True, keep_vba=False)
            return cache[name].cell(labels[name][label], 2).value
        river = {
            "length_m": val("River", "ChannelLength:"),
            "cells": val("River", "NumberOfCells:"),
            "width_m": val("River", "RiverWidth:"),
            "depth_m": val("River", "WaterDepth:"),
            "slope_percent": val("River", "RiverSlope:"),
            "manning_coefficient": val("River", "ManningCoef(n):"),
            "flow_rate_m3_s": val("River", "Discharge:"),
            "velocity_m_s": val("River", "FlowVelocity:"),
            "diffusivity_m2_s": val("River", "Diffusivity:"),
        }
        if isinstance(river["velocity_m_s"], str) and river["velocity_m_s"].replace("$", "").upper() == "=B8/(B4*B5)":
            river["velocity_m_s"] = _number(river["flow_rate_m3_s"]) / (_number(river["width_m"]) * _number(river["depth_m"]))
            river["velocity_mode"] = "flow_formula"
        else:
            river["velocity_mode"] = "direct"
            if isinstance(river["velocity_m_s"], str) and river["velocity_m_s"].startswith("="):
                river["velocity_m_s"] = cached("River", "FlowVelocity:")
        atmosphere = {
            "air_temperature_c": val("Atmosphere", "AirTemperature:"),
            "wind_speed_m_s": val("Atmosphere", "WindSpeed:"),
            "minimum_heat_transfer_coefficient": val("Atmosphere", "h_min:", default=6.9),
            "h_min_label_present": "h_min:" in labels["Atmosphere"],
            "humidity_percent": val("Atmosphere", "AirHumidity:"),
            "solar_constant_w_m2": val("Atmosphere", "SolarConstant:"),
            "latitude_degrees": val("Atmosphere", "Latitude:"),
            "sky_temperature_c": val("Atmosphere", "SkyTemperature:"),
            "sky_temperature_imposed": "SkyTemperature:" in labels["Atmosphere"],
            "cloud_cover_percent": val("Atmosphere", "CloudCover:"),
            "sunrise_hour": val("Atmosphere", "SunRizeHour:"),
            "sunset_hour": val("Atmosphere", "SunSetHour:"),
            "o2_partial_pressure_bar": val("Atmosphere", "O2PartialPressure:"),
            "co2_partial_pressure_bar": val("Atmosphere", "CO2PartialPressure:"),
            "o2_pressure_label_present": "O2PartialPressure:" in labels["Atmosphere"],
            "co2_pressure_label_present": "CO2PartialPressure:" in labels["Atmosphere"],
            "henry_constants": [],
            "henry_header_order": ["O2", "CO2"],
        }
        pp = atmosphere["co2_partial_pressure_bar"]
        if isinstance(pp, str) and pp.replace("$", "").upper() == "=B15*0.3/159":
            atmosphere["co2_partial_pressure_bar"] = _number(atmosphere["o2_partial_pressure_bar"]) * 0.3 / 159
            atmosphere["co2_pressure_mode"] = "o2_formula"
        else:
            atmosphere["co2_pressure_mode"] = "direct"
            if isinstance(pp, str) and pp.startswith("="):
                atmosphere["co2_partial_pressure_bar"] = cached("Atmosphere", "CO2PartialPressure:")
        henry_row = labels["Atmosphere"].get("HenryConstants:")
        if henry_row is not None:
            headers = [sheets["Atmosphere"].cell(henry_row, col).value for col in (3, 4)]
            atmosphere["henry_header_order"] = headers
            gas_columns = {label: col for col, label in zip((3, 4), headers) if label in ("O2", "CO2")}
            row = henry_row + 1
            while row <= min(sheets["Atmosphere"].max_row, 118):
                temp = sheets["Atmosphere"].cell(row, 2).value
                if temp is None:
                    break
                atmosphere["henry_constants"].append({
                    "temperature_c": temp,
                    "o2": sheets["Atmosphere"].cell(row, gas_columns["O2"]).value if "O2" in gas_columns else None,
                    "co2": sheets["Atmosphere"].cell(row, gas_columns["CO2"]).value if "CO2" in gas_columns else None,
                })
                row += 1
        discharge_sheet = sheets["Discharges"]
        discharge_labels = labels["Discharges"]
        id_row = discharge_labels.get("DischargeNumbers:")
        ids = _column_values(discharge_sheet, id_row)
        discharges = []
        for col, id_value in enumerate(ids, start=2):
            if _number(id_value) <= 0:
                break
            item = {"id": id_value, "name": val("Discharges", "DischargeNames:", col),
                    "cell": val("Discharges", "DischargeCells:", col),
                    "location_km": val("Discharges", "DischargeLocations(km):", col),
                    "flow_rate_m3_s": val("Discharges", "DischargeFlowRates(m3/s):", col)}
            for prop, label in DISCHARGE_LABELS.items():
                item[prop] = val("Discharges", label, col)
            discharges.append(item)
        graph_sheet = sheets["Graph"]
        graph = {
            "time_series_cells": _graph_values(graph_sheet, 2),
            "longitudinal_profiles_percent": _graph_values(graph_sheet, 3),
            "no_major_gridlines": graph_sheet.cell(4, 1).value == "NoMajorGridlines:",
            "sheet_order": list(wb.sheetnames),
        }
        return {
            "main": {
                "simulation_duration_days": val("Main", "SimulationDuration(Days):"),
                "time_step_seconds": val("Main", "TimeStep(Seconds):"),
                "print_interval_seconds": val("Main", "Dtprint(seconds):"),
                "time_method": val("Main", "TimeDiscretisation:"),
                "advection": _yes(val("Main", "Advection:")),
                "advection_type": val("Main", "AdvectionType:"),
                "quick_up_ratio": val("Main", "QUICK_UP_Ratio:"),
                "quick_up_ratio_label_present": "QUICK_UP_Ratio:" in labels["Main"],
                "diffusion": _yes(val("Main", "Diffusion:")),
            },
            "river": river,
            "atmosphere": atmosphere,
            "discharges": discharges,
            "discharge_cells_label_present": "DischargeCells:" in discharge_labels,
            "discharge_property_labels_present": {name: label in discharge_labels for name, label in DISCHARGE_LABELS.items()},
            "properties": {name: _read_property(sheets[name], int(river["cells"])) for name in PROPERTIES},
            "graph": graph,
        }
    finally:
        wb.close()
        if cache is not None:
            cache.close()
