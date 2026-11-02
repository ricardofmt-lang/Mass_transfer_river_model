"""Local Streamlit front end for the river transport VBA replica.

Run with ``streamlit run app.py`` (or ``run.cmd``). The editor starts with
empty numerical fields; a saved JSON file or the bundled workbook example can
fill them in. The JSON structure is the normalized ``read_workbook`` schema.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit reruns this file without restarting the Python process. Reload an
# older in-memory solver once so a live session keeps its unsaved widget values
# while gaining structured numerical errors from the updated source files.
_loaded_solver = sys.modules.get("transportmodel.numerical")
_loaded_sources = sys.modules.get("transportmodel.sources")
if _loaded_solver is not None and (
    not hasattr(_loaded_solver, "SimulationError")
    or _loaded_sources is not None and not hasattr(_loaded_sources, "CellCalculationError")
):
    if _loaded_sources is not None:
        importlib.reload(_loaded_sources)
    importlib.reload(_loaded_solver)

from transportmodel.charts import COLOR_INDEX, COLOR_RGB, build_charts
from transportmodel.numerical import SimulationResult, run_model
from transportmodel.visualization3d import build_river_3d_figure
from transportmodel.workbook import PROPERTIES


ROOT = Path(__file__).resolve().parent
EXAMPLE_PATHS = (
    ROOT / "examples" / "river_example.json",
    ROOT / "tests" / "fixtures" / "excel_oracle" / "baseline_20.json",
)


def _wide(widget: Any) -> dict[str, Any]:
    """Use the current Streamlit width API while retaining 1.40 compatibility."""
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def blank_config() -> dict[str, Any]:
    """A complete editable schema with no numerical values pre-entered."""
    property_blank = {
        "name_cell": "",
        "name_label_matches_vba": False,
        "units": "",
        "active": False,
        "active_label_present": True,
        "default": None,
        "maximum": None,
        "maximum_label_present": True,
        "minimum": None,
        "minimum_label_present": True,
        "cyclic": False,
        "cyclic_label_present": False,
        "left_boundary": None,
        "right_boundary": None,
        "free_surface_flux": False,
        "sensible_heat": True,
        "latent_heat": True,
        "radiative_heat": True,
        "anaerobic_respiration": False,
        "bod_oxygen_half_saturation": None,
        "bod_logistic_max": None,
        "bod_logistic_max_label_present": True,
        "bod_growth_per_day": None,
        "bod_decay_per_day": None,
        "bod_decay_label_present": True,
        "generic_rate": None,
        "generic_rate_units": None,
        "generic_rate_label_present": True,
        "initial": {
            "type": None,
            "default_cell_value": None,
            "default_cell_label_present": True,
            "cells": [],
            "interval_kind": "CELL",
            "intervals": [],
        },
    }
    return {
        "main": {
            "simulation_duration_days": None,
            "time_step_seconds": None,
            "print_interval_seconds": None,
            "time_method": None,
            "advection": False,
            "advection_type": None,
            "quick_up_ratio": None,
            "quick_up_ratio_label_present": True,
            "diffusion": False,
        },
        "river": {
            "length_m": None,
            "cells": None,
            "width_m": None,
            "depth_m": None,
            "slope_percent": None,
            "manning_coefficient": None,
            "flow_rate_m3_s": None,
            "velocity_m_s": None,
            "velocity_mode": None,
            "diffusivity_m2_s": None,
        },
        "atmosphere": {
            "air_temperature_c": None,
            "wind_speed_m_s": None,
            "minimum_heat_transfer_coefficient": None,
            "h_min_label_present": True,
            "humidity_percent": None,
            "solar_constant_w_m2": None,
            "latitude_degrees": None,
            "sky_temperature_c": None,
            "sky_temperature_imposed": False,
            "cloud_cover_percent": None,
            "sunrise_hour": None,
            "sunset_hour": None,
            "o2_partial_pressure_bar": None,
            "o2_pressure_label_present": True,
            "co2_partial_pressure_bar": None,
            "co2_pressure_mode": None,
            "co2_pressure_label_present": True,
            "henry_constants": [],
            "henry_header_order": ["O2", "CO2"],
        },
        "discharges": [],
        "discharge_cells_label_present": True,
        "discharge_property_labels_present": {name: True for name in PROPERTIES},
        "properties": {name: deepcopy(property_blank) for name in PROPERTIES},
        "graph": {
            "time_series_cells": [],
            "longitudinal_profiles_percent": [],
            "no_major_gridlines": False,
        },
    }


def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Preserve any workbook metadata while filling absent schema fields."""
    result = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _native(value: Any) -> Any:
    """Convert pandas/NumPy editor values into JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _native(value.item())
    return value


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        item = {key: _native(row.get(key)) for key in columns}
        if any(value not in (None, "") for value in item.values()):
            records.append(item)
    return records


def _ui_key(base: str) -> str:
    """Give each loaded input set distinct widget identities in a live browser.

    Streamlit's frontend can replay an old widget value after ``st.rerun``.
    Deleting its session-state key alone does not always reset that browser
    value. A new key generation makes the loaded JSON the only initial source.
    """
    generation = st.session_state.get("input_generation", 0)
    return base if generation == 0 else f"{base}__{generation}"


def _number(
    label: str,
    value: Any,
    key: str,
    *,
    integer: bool = False,
    help: str | None = None,
) -> int | float | None:
    if value is not None and value != "":
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            value = None
    else:
        value = None
    return st.number_input(
        label,
        value=value,
        step=1.0 if integer else 0.1,
        format="%.10g",
        key=_ui_key(f"ui_{key}"),
        help=help,
    )


def _select(label: str, options: list[str], value: Any, key: str) -> str | None:
    index = options.index(str(value)) if str(value) in options else None
    return st.selectbox(label, options, index=index, placeholder="Choose an option", key=_ui_key(f"ui_{key}"))


def _clear_widgets() -> None:
    for key in list(st.session_state):
        if key.startswith("result_"):
            del st.session_state[key]


def _load_config(payload: dict[str, Any]) -> None:
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError("JSON must contain an input configuration object")
    st.session_state["base_config"] = _merge(blank_config(), config)
    st.session_state.pop("result", None)
    st.session_state.pop("result_config", None)
    st.session_state.pop("simulation_error_report", None)
    _clear_widgets()
    st.session_state["input_generation"] = st.session_state.get("input_generation", 0) + 1


def _example_payload() -> dict[str, Any]:
    for path in EXAMPLE_PATHS:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("No bundled river example JSON was found")


def _toolbar() -> None:
    st.title("River transport model")
    st.caption("Local Python replica of the Channel transport workbook")
    upload_col, example_col, blank_col = st.columns([2, 1, 1])
    with upload_col:
        uploaded = st.file_uploader("Load saved inputs JSON", type=["json"], key=_ui_key("ui_upload"))
        if uploaded is not None and st.button("Apply JSON inputs", key="apply_json"):
            try:
                _load_config(json.loads(uploaded.getvalue().decode("utf-8-sig")))
            except (ValueError, UnicodeError) as exc:
                st.error(f"Cannot load JSON: {exc}")
            else:
                st.rerun()
    with example_col:
        st.write("")
        if st.button("Load workbook example", key="load_example", **_wide(st.button)):
            try:
                _load_config(_example_payload())
            except (OSError, ValueError) as exc:
                st.error(f"Cannot load example: {exc}")
            else:
                st.rerun()
    with blank_col:
        st.write("")
        if st.button("Clear inputs", key="clear_inputs", **_wide(st.button)):
            _load_config(blank_config())
            st.rerun()


def _main_controls(config: dict[str, Any]) -> None:
    data = config["main"]
    st.subheader("Simulation controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["simulation_duration_days"] = _number(
            "Simulation duration (days)", data.get("simulation_duration_days"), "duration"
        )
    with c2:
        data["time_step_seconds"] = _number(
            "Time step (seconds)", data.get("time_step_seconds"), "time_step"
        )
    with c3:
        data["print_interval_seconds"] = _number(
            "Output interval (seconds)", data.get("print_interval_seconds"), "print_interval"
        )
    data["time_method"] = _select(
        "Time discretization", ["exp", "imp", "semi"],
        str(data["time_method"]).lower() if data.get("time_method") else None,
        "time_method",
    )
    st.caption("exp = explicit · imp = implicit · semi = Crank–Nicolson")
    c1, c2 = st.columns(2)
    with c1:
        data["advection"] = st.checkbox("Advection", value=bool(data.get("advection")), key=_ui_key("ui_advection"))
    with c2:
        data["diffusion"] = st.checkbox("Diffusion", value=bool(data.get("diffusion")), key=_ui_key("ui_diffusion"))
    if data["advection"]:
        data["advection_type"] = _select(
            "Advection method",
            ["upwind", "central", "quick", "quick_up"],
            str(data["advection_type"]).lower() if data.get("advection_type") else None,
            "advection_type",
        )
        if data["advection_type"] == "quick_up":
            data["quick_up_ratio"] = _number(
                "QUICK_UP gradient ratio", data.get("quick_up_ratio"), "quick_up_ratio",
                help="The workbook recommends a value between 2 and 10.",
            )
            with st.expander("Workbook QUICK_UP label"):
                data["quick_up_ratio_label_present"] = st.checkbox(
                    "QUICK_UP ratio label present",
                    value=bool(data.get("quick_up_ratio_label_present", True)),
                    key=_ui_key("ui_quick_up_ratio_label"),
                )


def _river(config: dict[str, Any]) -> None:
    data = config["river"]
    st.subheader("River geometry and flow")
    c1, c2 = st.columns(2)
    with c1:
        data["length_m"] = _number("Channel length (m)", data.get("length_m"), "length")
        data["width_m"] = _number("River width (m)", data.get("width_m"), "width")
        data["slope_percent"] = _number(
            "River slope (spreadsheet value)", data.get("slope_percent"), "slope",
            help="Enter the value as in the River worksheet; the macro divides this entry by 100.",
        )
        data["manning_coefficient"] = _number(
            "Manning coefficient n", data.get("manning_coefficient"), "manning",
            help="Shown in the workbook; the Channel macro does not use it in transport calculations.",
        )
    with c2:
        data["cells"] = _number("Number of cells", data.get("cells"), "cells", integer=True)
        data["depth_m"] = _number("Water depth (m)", data.get("depth_m"), "depth")
        data["flow_rate_m3_s"] = _number(
            "River flow rate (m³/s)", data.get("flow_rate_m3_s"), "flow_rate"
        )
    velocity_choice = (
        "Calculate from flow rate" if data.get("velocity_mode") == "flow_formula"
        else "Enter velocity" if data.get("velocity_mode") == "direct" else None
    )
    mode_label = _select(
        "Flow velocity", ["Enter velocity", "Calculate from flow rate"],
        velocity_choice, "velocity_mode",
    )
    data["velocity_mode"] = (
        "flow_formula" if mode_label == "Calculate from flow rate"
        else "direct" if mode_label == "Enter velocity" else None
    )
    if data["velocity_mode"] == "direct":
        data["velocity_m_s"] = _number(
            "Flow velocity (m/s)", data.get("velocity_m_s"), "velocity"
        )
    elif data["velocity_mode"] == "flow_formula":
        q, width, depth = data.get("flow_rate_m3_s"), data.get("width_m"), data.get("depth_m")
        data["velocity_m_s"] = q / (width * depth) if q is not None and width and depth else None
        if data["velocity_m_s"] is not None:
            st.caption(f"Calculated velocity: {data['velocity_m_s']:.10g} m/s")
    diffusivity_choice = (
        "Standard estimate"
        if str(data.get("diffusivity_m2_s")) in ("std", "standard", "Standard", "STANDARD")
        else "Enter diffusivity" if data.get("diffusivity_m2_s") is not None else None
    )
    diffusivity_mode = _select(
        "Longitudinal diffusivity", ["Enter diffusivity", "Standard estimate"],
        diffusivity_choice, "diffusivity_mode",
    )
    if diffusivity_mode == "Standard estimate":
        data["diffusivity_m2_s"] = "std"
        if data.get("velocity_m_s") is not None and data.get("width_m") is not None:
            estimate = 0.1 * abs(data["velocity_m_s"]) * data["width_m"]
            st.caption(f"Standard estimate: {estimate:.10g} m²/s")
    elif diffusivity_mode == "Enter diffusivity":
        prior = data.get("diffusivity_m2_s")
        data["diffusivity_m2_s"] = _number(
            "Diffusivity (m²/s)", None if isinstance(prior, str) else prior, "diffusivity"
        )


def _henry_editor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows, columns=["temperature_c", "o2", "co2"])
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        **_wide(st.data_editor),
        key=_ui_key("ui_henry"),
        column_config={
            "temperature_c": st.column_config.NumberColumn("Water temperature (°C)"),
            "o2": st.column_config.NumberColumn("O₂ Henry constant (M/atm)"),
            "co2": st.column_config.NumberColumn("CO₂ Henry constant (M/atm)"),
        },
    )
    return _records(edited, ["temperature_c", "o2", "co2"])


def _atmosphere(config: dict[str, Any]) -> None:
    data = config["atmosphere"]
    st.subheader("Atmosphere")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["air_temperature_c"] = _number(
            "Air temperature (°C)", data.get("air_temperature_c"), "air_temp"
        )
        data["wind_speed_m_s"] = _number(
            "Wind speed (m/s)", data.get("wind_speed_m_s"), "wind"
        )
        data["minimum_heat_transfer_coefficient"] = _number(
            "Minimum heat transfer coefficient", data.get("minimum_heat_transfer_coefficient"), "h_min"
        )
        data["humidity_percent"] = _number(
            "Relative humidity (%)", data.get("humidity_percent"), "humidity"
        )
    with c2:
        data["solar_constant_w_m2"] = _number(
            "Solar constant (W/m²)", data.get("solar_constant_w_m2"), "solar_constant"
        )
        data["latitude_degrees"] = _number(
            "Latitude (degrees)", data.get("latitude_degrees"), "latitude"
        )
        data["cloud_cover_percent"] = _number(
            "Cloud cover (%)", data.get("cloud_cover_percent"), "cloud"
        )
    with c3:
        data["sunrise_hour"] = _number("Sunrise hour", data.get("sunrise_hour"), "sunrise")
        data["sunset_hour"] = _number("Sunset hour", data.get("sunset_hour"), "sunset")
        data["sky_temperature_imposed"] = st.checkbox(
            "Specify sky temperature", value=bool(data.get("sky_temperature_imposed")), key=_ui_key("ui_sky_imposed")
        )
        if data["sky_temperature_imposed"]:
            data["sky_temperature_c"] = _number(
                "Sky temperature (°C)", data.get("sky_temperature_c"), "sky_temp"
            )
    st.markdown("#### Atmospheric gas pressures")
    data["o2_partial_pressure_bar"] = _number(
        "O₂ partial pressure (workbook units)", data.get("o2_partial_pressure_bar"), "o2_pressure"
    )
    co2_choice = (
        "Calculate from O₂ pressure" if data.get("co2_pressure_mode") == "o2_formula"
        else "Enter directly" if data.get("co2_pressure_mode") == "direct" else None
    )
    pressure_mode = _select(
        "CO₂ partial pressure", ["Enter directly", "Calculate from O₂ pressure"],
        co2_choice, "co2_mode",
    )
    data["co2_pressure_mode"] = (
        "o2_formula" if pressure_mode == "Calculate from O₂ pressure"
        else "direct" if pressure_mode == "Enter directly" else None
    )
    if data["co2_pressure_mode"] == "o2_formula":
        o2 = data.get("o2_partial_pressure_bar")
        data["co2_partial_pressure_bar"] = o2 * 0.3 / 159 if o2 is not None else None
        if data["co2_partial_pressure_bar"] is not None:
            st.caption(f"Calculated CO₂ pressure: {data['co2_partial_pressure_bar']:.12g}")
    elif data["co2_pressure_mode"] == "direct":
        data["co2_partial_pressure_bar"] = _number(
            "CO₂ partial pressure (workbook units)", data.get("co2_partial_pressure_bar"), "co2_pressure"
        )
    st.markdown("#### Henry constants")
    st.caption("Add one row per measured temperature. The macro linearly interpolates between rows.")
    order = data.get("henry_header_order") or ["O2", "CO2"]
    order_label = st.selectbox(
        "Henry gas column order",
        ["O₂ then CO₂", "CO₂ then O₂"],
        index=1 if order == ["CO2", "O2"] else 0,
        key=_ui_key("ui_henry_order"),
    )
    data["henry_header_order"] = ["CO2", "O2"] if order_label == "CO₂ then O₂" else ["O2", "CO2"]
    data["henry_constants"] = _henry_editor(data.get("henry_constants") or [])
    with st.expander("Workbook atmosphere label switches"):
        data["h_min_label_present"] = st.checkbox(
            "Minimum heat coefficient label present",
            value=bool(data.get("h_min_label_present", True)), key=_ui_key("ui_h_min_label"),
        )
        data["o2_pressure_label_present"] = st.checkbox(
            "O₂ pressure label present",
            value=bool(data.get("o2_pressure_label_present", True)), key=_ui_key("ui_o2_pressure_label"),
        )
        data["co2_pressure_label_present"] = st.checkbox(
            "CO₂ pressure label present",
            value=bool(data.get("co2_pressure_label_present", True)), key=_ui_key("ui_co2_pressure_label"),
        )


def _discharge_editor(config: dict[str, Any]) -> None:
    st.subheader("Continuous lateral discharges")
    config["discharge_cells_label_present"] = _select(
        "Locate each discharge by", ["Cell number", "Distance (km)"],
        "Cell number" if config.get("discharge_cells_label_present", True) else "Distance (km)",
        "discharge_location_mode",
    ) == "Cell number"
    location = "cell" if config["discharge_cells_label_present"] else "location_km"
    visible_columns = ["id", "name", location, "flow_rate_m3_s", *PROPERTIES]
    frame = pd.DataFrame(config.get("discharges") or [], columns=visible_columns)
    column_config: dict[str, Any] = {
        "id": st.column_config.NumberColumn("Number", step=1),
        "name": st.column_config.TextColumn("Name"),
        "cell": st.column_config.NumberColumn("Cell", step=1),
        "location_km": st.column_config.NumberColumn("Distance (km)"),
        "flow_rate_m3_s": st.column_config.NumberColumn("Flow rate (m³/s)"),
        "Temperature": st.column_config.NumberColumn("Temperature (°C)"),
        "DO": st.column_config.NumberColumn("DO (mg/L)"),
        "BOD": st.column_config.NumberColumn("BOD (mg/L)"),
        "CO2": st.column_config.NumberColumn("CO₂ (mg/L)"),
        "Generic": st.column_config.NumberColumn("Generic"),
    }
    edited = st.data_editor(
        frame, num_rows="dynamic", hide_index=True, **_wide(st.data_editor),
        key=_ui_key(f"ui_discharges_{location}"), column_config=column_config,
    )
    raw = _records(edited, visible_columns)
    old = config.get("discharges") or []
    discharges: list[dict[str, Any]] = []
    for i, row in enumerate(raw, start=1):
        item = deepcopy(old[i - 1]) if i <= len(old) else {}
        item.update(row)
        if item.get("id") is None:
            item["id"] = i
        discharges.append(item)
    config["discharges"] = discharges
    with st.expander("Workbook discharge label switches"):
        st.caption("Uncheck a property to omit its discharge data label, as in the workbook.")
        labels = config.get("discharge_property_labels_present") or {}
        for name in PROPERTIES:
            labels[name] = st.checkbox(
                f"{name} discharge values present",
                value=bool(labels.get(name, True)), key=_ui_key(f"ui_discharge_label_{name}"),
            )
        config["discharge_property_labels_present"] = labels


def _property_editor(name: str, config: dict[str, Any]) -> None:
    data = config["properties"][name]
    st.subheader(f"{name} property")
    data["active"] = st.checkbox(
        f"Simulate {name}", value=bool(data.get("active")), key=_ui_key(f"ui_{name}_active")
    )
    if not data["active"]:
        with st.expander("Workbook activation label"):
            data["active_label_present"] = st.checkbox(
                "Property active label present", value=bool(data.get("active_label_present", True)),
                key=_ui_key(f"ui_{name}_active_label"),
            )
    if not data["active"]:
        st.caption("Activate this property to edit its model inputs.")
        return
    c1, c2 = st.columns(2)
    with c1:
        data["name_cell"] = st.text_input(
            "Display name in worksheet", value=str(data.get("name_cell") or ""), key=_ui_key(f"ui_{name}_name")
        )
        data["units"] = st.text_input(
            "Property units", value=str(data.get("units") or ""), key=_ui_key(f"ui_{name}_units")
        )
        data["default"] = _number("Default value", data.get("default"), f"{name}_default")
        data["minimum"] = _number("Minimum value", data.get("minimum"), f"{name}_minimum")
        data["maximum"] = _number("Maximum value", data.get("maximum"), f"{name}_maximum")
    with c2:
        if config["main"].get("advection"):
            data["left_boundary"] = _number(
                "Left boundary value", data.get("left_boundary"), f"{name}_left"
            )
            data["right_boundary"] = _number(
                "Right boundary value", data.get("right_boundary"), f"{name}_right"
            )
        if name == "Temperature":
            data["cyclic"] = st.checkbox(
                "Cyclic temperature boundary", value=bool(data.get("cyclic")), key=_ui_key(f"ui_{name}_cyclic")
            )
        if name in ("Temperature", "DO", "CO2"):
            data["free_surface_flux"] = st.checkbox(
                "Exchange with atmosphere", value=bool(data.get("free_surface_flux")),
                key=_ui_key(f"ui_{name}_surface"),
            )
        if name == "Temperature" and data["free_surface_flux"]:
            data["sensible_heat"] = st.checkbox(
                "Sensible heat", value=bool(data.get("sensible_heat", True)), key=_ui_key("ui_heat_sensible")
            )
            data["latent_heat"] = st.checkbox(
                "Latent heat", value=bool(data.get("latent_heat", True)), key=_ui_key("ui_heat_latent")
            )
            data["radiative_heat"] = st.checkbox(
                "Long-wave radiation", value=bool(data.get("radiative_heat", True)), key=_ui_key("ui_heat_radiative")
            )
            st.caption("Solar radiation runs whenever atmospheric heat exchange is enabled.")

    if name == "Generic":
        st.markdown("#### Decay or growth")
        c1, c2 = st.columns(2)
        with c1:
            data["generic_rate"] = _number(
                "Decay / growth rate", data.get("generic_rate"), "generic_rate"
            )
        with c2:
            data["generic_rate_units"] = _select(
                "Rate units", ["T90(Hours)", "HalfLife(Days)", "TDuplicate(Days)", "Day-1", "RT"],
                data.get("generic_rate_units"), "generic_rate_units",
            )
    if name == "BOD":
        st.markdown("#### BOD reaction")
        c1, c2 = st.columns(2)
        with c1:
            data["bod_decay_per_day"] = _number(
                "Decay rate (day⁻¹)", data.get("bod_decay_per_day"), "bod_decay"
            )
            data["bod_growth_per_day"] = _number(
                "Logistic growth rate (day⁻¹)", data.get("bod_growth_per_day"), "bod_growth"
            )
        with c2:
            data["bod_logistic_max"] = _number(
                "Logistic maximum (mg/L)", data.get("bod_logistic_max"), "bod_max"
            )
            data["bod_oxygen_half_saturation"] = _number(
                "O₂ half-saturation (mg/L)", data.get("bod_oxygen_half_saturation"), "bod_ksat"
            )
        data["anaerobic_respiration"] = st.checkbox(
            "Anaerobic respiration", value=bool(data.get("anaerobic_respiration")), key=_ui_key("ui_bod_anaerobic")
        )

    st.markdown("#### Initial distribution")
    initial = data["initial"]
    initial["type"] = _select(
        "Initialization", ["DEFAULT", "CELL", "INTERVAL"], initial.get("type"),
        f"{name}_initial_type",
    )
    if initial["type"] is not None:
        initial["default_cell_value"] = _number(
            "Initial background value", initial.get("default_cell_value"),
            f"{name}_background",
            help="The workbook fills every cell with this value before any cell or interval entries.",
        )
    if initial["type"] == "CELL":
        frame = pd.DataFrame(initial.get("cells") or [], columns=["cell", "value"])
        edited = st.data_editor(
            frame, num_rows="dynamic", hide_index=True, **_wide(st.data_editor),
            key=_ui_key(f"ui_{name}_initial_cells"),
            column_config={
                "cell": st.column_config.NumberColumn("Cell number", step=1),
                "value": st.column_config.NumberColumn("Initial value"),
            },
        )
        initial["cells"] = _records(edited, ["cell", "value"])
    elif initial["type"] == "INTERVAL":
        kind = _select(
            "Interval coordinates", ["CELL", "m"], initial.get("interval_kind"),
            f"{name}_interval_kind",
        )
        initial["interval_kind"] = kind
        frame = pd.DataFrame(initial.get("intervals") or [], columns=["start", "end", "value"])
        edited = st.data_editor(
            frame, num_rows="dynamic", hide_index=True, **_wide(st.data_editor),
            key=_ui_key(f"ui_{name}_initial_intervals"),
            column_config={
                "start": st.column_config.NumberColumn("First cell" if kind == "CELL" else "Start (m)"),
                "end": st.column_config.NumberColumn("Last cell" if kind == "CELL" else "End (m)"),
                "value": st.column_config.NumberColumn("Initial value"),
            },
        )
        initial["intervals"] = _records(edited, ["start", "end", "value"])

    with st.expander("Workbook compatibility options"):
        st.caption("These switches represent label presence in the spreadsheet and preserve its exact behavior.")
        data["active_label_present"] = st.checkbox(
            "Property active label present", value=bool(data.get("active_label_present", True)),
            key=_ui_key(f"ui_{name}_active_label"),
        )
        data["name_label_matches_vba"] = st.checkbox(
            "Property name label matches macro", value=bool(data.get("name_label_matches_vba")),
            key=_ui_key(f"ui_{name}_name_label"),
        )
        data["minimum_label_present"] = st.checkbox(
            "Minimum value label present", value=bool(data.get("minimum_label_present", True)),
            key=_ui_key(f"ui_{name}_minimum_label"),
        )
        data["maximum_label_present"] = st.checkbox(
            "Maximum value label present", value=bool(data.get("maximum_label_present", True)),
            key=_ui_key(f"ui_{name}_maximum_label"),
        )
        data["cyclic_label_present"] = st.checkbox(
            "Cyclic boundary label present", value=bool(data.get("cyclic_label_present")),
            key=_ui_key(f"ui_{name}_cyclic_label"),
        )
        initial["default_cell_label_present"] = st.checkbox(
            "Initial background value label present",
            value=bool(initial.get("default_cell_label_present", True)),
            key=_ui_key(f"ui_{name}_background_label"),
        )
        if name == "Generic":
            data["generic_rate_label_present"] = st.checkbox(
                "Generic rate label present",
                value=bool(data.get("generic_rate_label_present", True)),
                key=_ui_key("ui_generic_rate_label"),
            )
        if name == "BOD":
            data["bod_decay_label_present"] = st.checkbox(
                "BOD decay rate label present",
                value=bool(data.get("bod_decay_label_present", True)),
                key=_ui_key("ui_bod_decay_label"),
            )
            data["bod_logistic_max_label_present"] = st.checkbox(
                "BOD logistic maximum label present",
                value=bool(data.get("bod_logistic_max_label_present", True)),
                key=_ui_key("ui_bod_max_label"),
            )
        if name != "Temperature":
            data["cyclic"] = st.checkbox(
                "Cyclic boundary value (unused for this property)",
                value=bool(data.get("cyclic")), key=_ui_key(f"ui_{name}_unused_cyclic"),
            )
        if name in ("Generic", "BOD"):
            data["free_surface_flux"] = st.checkbox(
                "Free-surface label value (unused for this property)",
                value=bool(data.get("free_surface_flux")), key=_ui_key(f"ui_{name}_unused_surface"),
            )


def _graph_editor(config: dict[str, Any]) -> None:
    graph = config["graph"]
    st.subheader("Graph selections")
    if st.button("Use standard graph selection", key="graph_defaults"):
        cells = config["river"].get("cells")
        last = int(cells) if cells is not None and cells >= 1 else 1
        graph["time_series_cells"] = list(dict.fromkeys([1, max(1, (last + 1) // 2), last]))
        graph["longitudinal_profiles_percent"] = [0, 50, 100]
        st.session_state["base_config"]["graph"] = deepcopy(graph)
        for key in ("ui_graph_cells", "ui_graph_profiles"):
            st.session_state.pop(_ui_key(key), None)
        st.rerun()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Time series at cells**")
        frame = pd.DataFrame(
            [{"cell": value} for value in graph.get("time_series_cells") or []], columns=["cell"]
        )
        edited = st.data_editor(
            frame, num_rows="dynamic", hide_index=True, **_wide(st.data_editor),
            key=_ui_key("ui_graph_cells"),
            column_config={"cell": st.column_config.NumberColumn("Cell number", step=1)},
        )
        graph["time_series_cells"] = [row["cell"] for row in _records(edited, ["cell"])]
    with c2:
        st.markdown("**Longitudinal profiles at time percentages**")
        frame = pd.DataFrame(
            [{"percent": value} for value in graph.get("longitudinal_profiles_percent") or []],
            columns=["percent"],
        )
        edited = st.data_editor(
            frame, num_rows="dynamic", hide_index=True, **_wide(st.data_editor),
            key=_ui_key("ui_graph_profiles"),
            column_config={"percent": st.column_config.NumberColumn("Simulation time (%)")},
        )
        graph["longitudinal_profiles_percent"] = [
            row["percent"] for row in _records(edited, ["percent"])
        ]
    graph["no_major_gridlines"] = st.checkbox(
        "Hide major gridlines", value=bool(graph.get("no_major_gridlines")),
        key=_ui_key("ui_graph_no_grid"),
    )
    if graph.get("sheet_order"):
        with st.expander("Worksheet chart order"):
            text = st.text_area(
                "Sheet names, one per line", value="\n".join(map(str, graph["sheet_order"])),
                key=_ui_key("ui_graph_sheet_order"),
            )
            graph["sheet_order"] = [line.strip() for line in text.splitlines() if line.strip()]


def _result_table(result: SimulationResult, name: str) -> pd.DataFrame:
    columns = {"Time (s)": result.time_seconds,
               "Time (days)": [time / 86400 for time in result.time_seconds]}
    for i, x in enumerate(result.coordinates_m, start=1):
        columns[f"Cell {i} · {x:.6g} m"] = [row[i - 1] for row in result.values[name]]
    return pd.DataFrame(columns)


def _simple_plots(result: SimulationResult, name: str) -> None:
    values = result.values[name]
    days = [time / 86400 for time in result.time_seconds]
    if not values:
        st.info("No printed values were returned for this property.")
        return
    profile_tab, series_tab, heatmap_tab = st.tabs(
        ["Longitudinal profile", "Cell time series", "Space–time map"]
    )
    with profile_tab:
        row = st.slider(
            "Printed time", min_value=0, max_value=len(values) - 1,
            value=len(values) - 1, key="result_profile_row",
        )
        color = COLOR_RGB.get(COLOR_INDEX.get(name, 33), "#00CCFF")
        fig = go.Figure(go.Scatter(x=result.coordinates_m, y=values[row], mode="lines+markers",
                                   line={"color": color}))
        fig.update_layout(
            title=f"{name} at {days[row]:.6g} days",
            xaxis_title="Distance to origin (m)", yaxis_title=name,
        )
        st.plotly_chart(fig, **_wide(st.plotly_chart))
    with series_tab:
        cell = st.number_input(
            "Cell number", min_value=1, max_value=len(result.coordinates_m),
            value=max(1, (len(result.coordinates_m) + 1) // 2), step=1,
            key="result_cell_number",
        )
        color = COLOR_RGB.get(COLOR_INDEX.get(name, 33), "#00CCFF")
        fig = go.Figure(go.Scatter(
            x=days, y=[row[cell - 1] for row in values], mode="lines+markers",
            line={"color": color},
        ))
        fig.update_layout(
            title=f"{name} in cell {cell}", xaxis_title="Time (days)", yaxis_title=name,
        )
        st.plotly_chart(fig, **_wide(st.plotly_chart))
    with heatmap_tab:
        fig = go.Figure(go.Heatmap(
            x=result.coordinates_m, y=days, z=values, colorscale="Turbo",
            colorbar={"title": name},
        ))
        fig.update_layout(
            title=f"{name} along the river and through time",
            xaxis_title="Distance to origin (m)", yaxis_title="Time (days)",
        )
        st.plotly_chart(fig, **_wide(st.plotly_chart))


def _macro_charts(result: SimulationResult, config: dict[str, Any]) -> None:
    st.caption("These charts use only the current Python simulation; Excel is not required.")
    try:
        charts = build_charts(result, config)
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        st.error(f"Graph selection cannot be rendered: {exc}")
        return
    if not charts:
        st.info("Add cell numbers or time percentages in Graph selections to create workbook-style charts.")
        return
    choices = [f"{i + 1}. {chart['title']} · {chart['y_axis_title']}" for i, chart in enumerate(charts)]
    selected = st.selectbox("Workbook-style graph", choices, key="result_macro_chart")
    chart = charts[choices.index(selected)]
    fig = go.Figure()
    for index, series in enumerate(chart["series"]):
        fig.add_trace(go.Scatter(
            x=series["x_values"], y=series["y_values"], name=series["name"],
            mode="lines", line={"shape": "spline" if chart["chart_type"] == 72 else "linear",
                                "color": chart["line_color_hex"] if index == len(chart["series"]) - 1 else None},
        ))
    fig.update_layout(
        title=chart["title"], xaxis_title=chart["x_axis_title"],
        yaxis_title=chart["y_axis_title"],
    )
    if chart["y_axis_minimum"] is not None and chart["y_axis_maximum"] is not None:
        if chart["y_axis_minimum"] < chart["y_axis_maximum"]:
            fig.update_yaxes(range=[chart["y_axis_minimum"], chart["y_axis_maximum"]])
    st.plotly_chart(fig, **_wide(st.plotly_chart))
    with st.expander("Chart data and workbook-style series formulas"):
        st.json(chart)


def _river_3d(result: SimulationResult, name: str, config: dict[str, Any]) -> None:
    river = config["river"]
    units = "kg" if name.startswith("Integrated") else (config.get("properties", {}).get(name) or {}).get("units")
    if units == "�C":  # The source workbook stores a replacement character here.
        units = "°C"
    fig = build_river_3d_figure(
        result, name,
        length_m=float(river["length_m"]),
        width_m=float(river["width_m"]),
        depth_m=float(river["depth_m"]),
        units=units,
    )
    st.plotly_chart(fig, key="result_river_3d", **_wide(st.plotly_chart))
    st.caption(
        f"Channel width: {float(river['width_m']):g} m; "
        f"depth: {float(river['depth_m']):g} m. "
        "Each modeled cell has one value across the full width and depth. "
        "The display shortens the river relative to its cross-section for readability; "
        "axis coordinates and hover values remain in actual metres. "
        "The color scale stays fixed across all printed times."
    )


def _results(result: SimulationResult, config: dict[str, Any]) -> None:
    st.header("Results")
    st.caption("Showing the last completed simulation.")
    diag = result.diagnostics
    keys = [
        ("Courant number", "courant_number"),
        ("Diffusion number", "diffusion_number"),
        ("Grid Reynolds number", "grid_reynolds_number"),
        ("River residence (days)", "residence_time_river_days"),
        ("Cell residence (s)", "residence_time_cell_seconds"),
    ]
    cols = st.columns(len(keys))
    for col, (label, key) in zip(cols, keys):
        col.metric(label, str(diag.get(key, "")))
    names = list(result.values)
    if not names:
        st.info("The run completed with no active properties.")
        return
    name = st.selectbox("Property or accumulated respiration", names, key="result_property")
    table_tab, charts_tab, river_3d_tab, workbook_tab = st.tabs(
        ["Numerical table", "Interactive graphs", "3D river animation", "Workbook graphs"]
    )
    with table_tab:
        table = _result_table(result, name)
        st.dataframe(table, height=440, **_wide(st.dataframe))
        st.download_button(
            "Download table CSV", table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"river_{name.replace('/', '_')}_results.csv", mime="text/csv",
            key="result_csv",
        )
    with charts_tab:
        _simple_plots(result, name)
    with river_3d_tab:
        _river_3d(result, name, config)
    with workbook_tab:
        _macro_charts(result, config)
    st.download_button(
        "Download all results JSON",
        json.dumps(_native(result.as_dict()), indent=2, allow_nan=False).encode("utf-8"),
        file_name="river_results.json", mime="application/json", key="result_json",
    )


def _simulation_error_report(exc: Exception) -> dict[str, str]:
    """Turn a solver failure into a readable, persistent Streamlit report."""
    original_type = str(getattr(exc, "original_type", type(exc).__name__))
    if original_type in ("OverflowError", "FloatingPointError"):
        summary = "A calculation produced a number too large for the computer to represent."
        default_hint = (
            "Check the values and time step used in this calculation. "
            "Extreme inputs can make exponential or power calculations overflow."
        )
    elif original_type == "ZeroDivisionError":
        summary = "A calculation attempted to divide by zero."
        default_hint = "Check zero-valued dimensions, flow, time step, and rate parameters."
    elif original_type in ("ValueError", "TypeError"):
        summary = "An input or calculation could not be evaluated."
        default_hint = "Review the inputs named below and run the simulation again."
    else:
        summary = "The simulation stopped because a calculation failed."
        default_hint = "Review the failure location and technical details below."

    detail = str(getattr(exc, "detail", "") or "").strip()
    if not detail:
        detail = str(exc).strip() or f"{original_type} raised without a message."
    if original_type == "OverflowError" and detail in {
        "math range error", "(34, 'Result too large')",
    }:
        detail = "A mathematical function exceeded the floating-point number range."

    context: list[str] = []
    stage = getattr(exc, "stage", None)
    property_name = getattr(exc, "property_name", None)
    step = getattr(exc, "step", None)
    time_seconds = getattr(exc, "time_seconds", None)
    cell = getattr(exc, "cell", None)
    if stage:
        context.append(f"Calculation: {stage}")
    if property_name:
        context.append(f"Property: {property_name}")
    if step is not None:
        context.append(f"Step: {step}")
    if time_seconds is not None:
        seconds = float(time_seconds)
        context.append(f"Time: {seconds:g} s ({seconds / 86400:.6g} days)")
    if cell is not None:
        context.append(f"River cell: {cell}")
    if not context:
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            frame = frames[-1]
            context.append(f"Function: {frame.name} ({Path(frame.filename).name}:{frame.lineno})")
    return {
        "summary": summary,
        "detail": detail,
        "where": " · ".join(context) if context else "Location unavailable",
        "hint": str(getattr(exc, "hint", None) or default_hint),
        "technical": "".join(traceback.format_exception(exc)),
    }


def _show_simulation_error(report: dict[str, str]) -> None:
    st.error(f"Simulation stopped: {report['summary']}")
    st.write(f"**What failed:** {report['detail']}")
    st.write(f"**Where:** {report['where']}")
    st.info(f"**What to check:** {report['hint']}")
    st.caption("This report is from the last attempted run. Run the simulation again after changing inputs.")
    with st.expander("Technical details"):
        st.code(report["technical"], language="text")


def main() -> None:
    st.set_page_config(page_title="River transport model", layout="wide")
    if "base_config" not in st.session_state:
        st.session_state["base_config"] = blank_config()
    _toolbar()
    config = deepcopy(st.session_state["base_config"])
    controls_tab, river_tab, atmosphere_tab, discharges_tab, properties_tab, graph_tab = st.tabs(
        ["Controls", "River", "Atmosphere", "Discharges", "Properties", "Graphs"]
    )
    with controls_tab:
        _main_controls(config)
    with river_tab:
        _river(config)
    with atmosphere_tab:
        _atmosphere(config)
    with discharges_tab:
        _discharge_editor(config)
    with properties_tab:
        property_tabs = st.tabs(list(PROPERTIES))
        for name, tab in zip(PROPERTIES, property_tabs):
            with tab:
                _property_editor(name, config)
    with graph_tab:
        _graph_editor(config)

    config = _native(config)
    st.session_state["base_config"] = config
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        run = st.button("Run simulation", type="primary", key="run_model", **_wide(st.button))
    with c2:
        st.download_button(
            "Download saved inputs JSON",
            json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            file_name="river_inputs.json", mime="application/json",
            key="download_config", **_wide(st.download_button),
        )
    if run:
        try:
            with st.spinner("Running river simulation…"):
                result = run_model(config)
        except Exception as exc:
            st.session_state.pop("result", None)
            st.session_state.pop("result_config", None)
            st.session_state["simulation_error_report"] = _simulation_error_report(exc)
        else:
            st.session_state.pop("simulation_error_report", None)
            st.session_state["result"] = result
            st.session_state["result_config"] = deepcopy(config)
            for key in list(st.session_state):
                if key.startswith("result_") and key != "result_config":
                    del st.session_state[key]
    if "simulation_error_report" in st.session_state:
        _show_simulation_error(st.session_state["simulation_error_report"])
    result = st.session_state.get("result")
    if result is not None:
        _results(result, st.session_state["result_config"])


if __name__ == "__main__":
    main()
