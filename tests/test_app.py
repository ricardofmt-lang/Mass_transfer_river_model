"""Exercise the actual Streamlit input-to-solver-to-visualization path."""

from __future__ import annotations

import json
import math
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
ORACLE = Path(__file__).parent / "fixtures" / "excel_oracle"


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    return at


def _input_key(at: AppTest, base: str) -> str:
    generation = at.session_state["input_generation"] if "input_generation" in at.session_state else 0
    return base if generation == 0 else f"{base}__{generation}"


def _matches_excel(result, case: str) -> None:
    fixture = json.loads((ORACLE / f"{case}.json").read_text(encoding="utf-8"))
    for name, output in fixture["outputs"].items():
        assert name in result.values
        for i, (expected_row, actual_row) in enumerate(zip(output["data"], result.values[name])):
            assert math.isclose(expected_row[0], result.time_seconds[i])
            for expected, actual in zip(expected_row[3:], actual_row):
                assert math.isclose(expected, actual, rel_tol=1e-14, abs_tol=1e-12)


def test_blank_start_example_run_and_visualization() -> None:
    at = _app()
    assert all(widget.value is None for widget in at.number_input[:3])
    assert at.selectbox(key="ui_time_method").value is None
    assert "result" not in at.session_state
    assert len(at.get("plotly_chart")) == 0
    at.button(key="load_example").click().run()
    assert not at.exception
    assert at.session_state["input_generation"] == 1
    assert at.checkbox(key=_input_key(at, "ui_Generic_active")).value is True
    assert at.checkbox(key=_input_key(at, "ui_Temperature_active")).value is True
    assert [widget.value for widget in at.number_input[:3]] == [0.005, 60.0, 120.0]
    at.button(key="run_model").click().run()
    assert not at.exception
    _matches_excel(at.session_state["result"], "baseline_20")
    assert len(at.dataframe) >= 1  # numerical table and editable input grids
    assert len(at.get("plotly_chart")) == 5  # profile, time series, map, 3D river, workbook chart
    at.button(key="clear_inputs").click().run()
    assert not at.exception
    assert at.session_state["input_generation"] == 2
    assert at.checkbox(key=_input_key(at, "ui_Generic_active")).value is False
    assert at.checkbox(key=_input_key(at, "ui_Temperature_active")).value is False
    assert all(widget.value is None for widget in at.number_input[:3])
    assert "result" not in at.session_state


def test_editing_a_heat_flux_switch_changes_excel_parity() -> None:
    at = _app()
    at.button(key="load_example").click().run()
    at.checkbox(key=_input_key(at, "ui_Temperature_surface")).set_value(False).run()
    assert not at.exception
    assert not at.session_state["base_config"]["properties"]["Temperature"]["free_surface_flux"]
    at.button(key="run_model").click().run()
    assert not at.exception
    _matches_excel(at.session_state["result"], "temperature_surface_off")


def test_property_initialization_controls_are_conditional() -> None:
    at = _app()
    at.button(key="load_example").click().run()
    at.selectbox(key=_input_key(at, "ui_Generic_initial_type")).set_value("CELL").run()
    assert at.session_state["base_config"]["properties"]["Generic"]["initial"]["type"] == "CELL"
    assert _input_key(at, "ui_Generic_initial_cells") in at.session_state
    assert _input_key(at, "ui_Generic_initial_intervals") not in at.session_state
    at.selectbox(key=_input_key(at, "ui_Generic_initial_type")).set_value("INTERVAL").run()
    assert at.session_state["base_config"]["properties"]["Generic"]["initial"]["type"] == "INTERVAL"
    assert _input_key(at, "ui_Generic_initial_intervals") in at.session_state
    assert not at.exception
