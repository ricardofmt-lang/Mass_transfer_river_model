"""Teaching cases exercise the same inputs and outputs as a normal run."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from transportmodel.charts import build_charts
from transportmodel.config import manning_velocity
from transportmodel.numerical import run_model
from transportmodel.presets import teaching_cases


CASES = {case.key: case for case in teaching_cases()}
APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("case", [case for case in CASES.values() if case.config], ids=lambda case: case.key)
def test_teaching_cases_run_and_produce_charts(case):
    result = run_model(case.config)
    assert result.time_seconds[0] == 0
    assert len(result.time_seconds) >= 2
    for rows in result.values.values():
        assert len(rows) == len(result.time_seconds)
        assert all(isfinite(value) for row in rows for value in row)
    assert build_charts(result, case.config)


def test_teaching_comparisons_have_the_intended_behaviour():
    def case(key, name="Generic"):
        return run_model(CASES[key].config).values[name]

    initial, final = case("four-cell-upwind")
    assert initial == [1, 1, 0, 0]
    assert final == [1, 1, 0.5, 0]
    assert case("four-cell-central")[-1] == [1, 1.25, 0.25, 0]
    assert case("decay-step-exp")[-1][0] == 90
    assert case("decay-step-imp")[-1][0] == pytest.approx(100 / 1.1)
    assert case("decay-step-semi")[-1][0] == pytest.approx(100 * 0.95 / 1.05)
    low = case("bod-low-wind", "DO")
    high = case("bod-high-wind", "DO")
    assert min(row[0] for row in low) < 0.1
    assert min(row[0] for row in high) > 4
    assert high[-1][0] > low[-1][0]
    baseline = case("daily-cycle", "Temperature")
    assert 19 < max(row[0] for row in baseline[-12:]) < 21


def test_manning_mode_calculates_velocity_from_percent_slope():
    result = manning_velocity({
        "width_m": 5, "depth_m": 0.5, "manning_coefficient": 0.025, "slope_percent": 0.01,
    })
    assert result == pytest.approx((2.5 / 6) ** (2 / 3) * 0.01 / 0.025)


def test_student_can_run_an_unstable_preset_and_edit_it():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.selectbox(key="case_topic").set_value("Numerical methods").run()
    at.selectbox(key="teaching_case_Numerical methods").set_value("central-unstable").run()
    at.button(key="load_example").click().run()
    assert at.session_state["loaded_case_name"] == "Unstable explicit central pulse"
    at.button(key="run_model").click().run()
    assert not at.exception
    assert any("outside its interior stability range" in item.value for item in at.warning)
    assert max(at.session_state["result"].values["Generic"][-1]) > 25
    at.number_input(key="ui_duration__1").set_value(0.0).run()
    assert any("Inputs have changed" in item.value for item in at.info)
    at.button(key="run_model").click().run()
    assert not at.exception
    assert at.session_state["result"].time_seconds == [0]
    assert at.get("plotly_chart")


def test_malformed_json_fails_with_field_context():
    from app import _merge, blank_config

    with pytest.raises(ValueError, match=r"properties.Generic.initial must be an object"):
        _merge(blank_config(), {"properties": {"Generic": {"initial": []}}})
    with pytest.raises(ValueError, match=r"Each row in discharges must be an object"):
        _merge(blank_config(), {"discharges": [12]})
