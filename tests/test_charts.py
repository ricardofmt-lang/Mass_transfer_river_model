"""Chart reconstruction against the recorded Excel macro oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transportmodel.charts import build_charts
from transportmodel.numerical import SimulationResult


BASELINE = Path(__file__).parent / "fixtures" / "excel_oracle" / "baseline_20.json"


def _result_from_oracle_outputs(fixture: dict) -> SimulationResult:
    outputs = fixture["outputs"]
    first = next(iter(outputs.values()))
    return SimulationResult(
        time_seconds=[row[0] for row in first["data"]],
        coordinates_m=first["headers"]["coordinates"][3:],
        values={name: [row[3:] for row in output["data"]]
                for name, output in outputs.items()},
    )


def test_baseline_charts_match_every_oracle_series_exactly() -> None:
    fixture = json.loads(BASELINE.read_text(encoding="utf-8"))
    charts = build_charts(_result_from_oracle_outputs(fixture), fixture["config"])
    assert len(charts) == len(fixture["graphs"]) == 22
    assert [
        {key: chart[key] for key in ("object_name", "title", "chart_type", "series")}
        for chart in charts
    ] == fixture["graphs"]
    assert [(chart["left"], chart["top"]) for chart in charts] == [
        (left, 100 + group * 350)
        for group in range(11) for left in (100, 650)
    ]
    assert all((chart["width"], chart["height"]) == (500, 325) for chart in charts)
    assert all(chart["chart_type"] == 74 for chart in charts[:4])
    assert all(chart["chart_type"] == 72 for chart in charts[4:])
    assert all(chart["x_axis_title"] == "Distance to origin (m)"
               for chart in charts[2:4] + charts[10:])
    assert [series["formula"].split(",")[-2] for series in charts[2]["series"]] == [
        "Temperature!$D$26:$X$26", "Temperature!$D$27:$X$27",
        "Temperature!$D$28:$X$28", "Temperature!$D$28:$X$28",
        "Temperature!$D$29:$X$29", "Temperature!$D$30:$X$30",
    ]


def _small_result() -> SimulationResult:
    return SimulationResult(
        time_seconds=[0.0, 10.0],
        coordinates_m=[5.0, 15.0, 25.0, 35.0],
        values={
            "Temperature": [[10, 11, 12, 13], [14, 15, 16, 17]],
            "DO": [[1, 2, 3, 4], [5, 6, 7, 8]],
        },
    )


def _small_config() -> dict:
    return {
        "properties": {"Temperature": {"active": True},
                       "DO": {"active": True}, "CO2": {"active": False}},
        "graph": {"time_series_cells": [1],
                  "longitudinal_profiles_percent": [100]},
    }


def test_charts_use_only_current_computed_properties_without_workbook_state() -> None:
    result = _small_result()
    config = _small_config()
    charts = build_charts(result, config)
    assert [chart["title"] for chart in charts[:2]] == [
        "Temperature vs time in cell ", "DO vs time in cell ",
    ]
    assert len(charts) == 8
    assert {chart["y_axis_title"] for chart in charts} == {"Temperature", "DO"}
    assert all(chart["source_status"] == "computed" for chart in charts)
    assert charts[1]["series"][0]["x_values"] == [0.0, 10 / 86400]
    assert charts[1]["series"][0]["y_values"] == [1, 5]

    # Prior Excel worksheet contents are irrelevant even if present in a
    # downloaded configuration from an older app version.
    config["existing_sheets"] = {
        "CO2": {"values": [[999] * 4] * 2, "time_days": [10, 20]}}
    assert build_charts(result, config) == charts


def test_active_co2_gets_charts_even_when_do_is_inactive() -> None:
    result = SimulationResult(
        time_seconds=[0, 2160], coordinates_m=[5, 15],
        values={"CO2": [[1, 2], [3, 4]]},
    )
    config = {"properties": {"DO": {"active": False}, "CO2": {"active": True}},
              "graph": {"time_series_cells": [1],
                        "longitudinal_profiles_percent": [100]}}
    charts = build_charts(result, config)
    assert len(charts) == 4
    assert all(chart["y_axis_title"] == "CO2" for chart in charts)
    assert charts[0]["series"][0]["y_values"] == [1, 3]
    assert charts[1]["series"][0]["y_values"] == [3, 4, None]
    assert charts[1]["series"][0]["name"] == " @Time 0.02 Days"
    assert charts[-1]["title"] == "Longitudinal Profile 0.02 days after start"


def test_cell_zero_and_invalid_later_cell_follow_macro_selection() -> None:
    config = _small_config()
    config["graph"] = {"time_series_cells": [0, 1, 5, 2],
                       "longitudinal_profiles_percent": []}
    charts = build_charts(_small_result(), config)
    assert len(charts) == 2 * 3  # two overlap cells and two individual rows
    assert charts[0]["series"][0]["formula"] == (
        '=SERIES(" @Cell 0",Temperature!$B$26:$B$27,Temperature!$C$26:$C$27,1)'
    )
    assert charts[0]["series"][0]["y_values"] == [None, None]
    assert charts[2]["title"] == "Time Series at Cell 0"


def test_blank_selection_inside_graph_row_becomes_zero() -> None:
    config = _small_config()
    config["graph"] = {"time_series_cells": [1, None, 3],
                       "longitudinal_profiles_percent": [None, 100]}
    charts = build_charts(_small_result(), config)
    assert [series["name"] for series in charts[0]["series"]] == [
        " @Cell 1", " @Cell 0", " @Cell 3",
    ]
    assert [series["name"] for series in charts[2]["series"]] == [
        " @Time 0 Days", " @Time 0 Days",
    ]


def test_space_only_starts_at_uninitialized_vba_position_zero() -> None:
    config = _small_config()
    config["graph"] = {"time_series_cells": [],
                       "longitudinal_profiles_percent": [100],
                       "no_major_gridlines": True}
    charts = build_charts(_small_result(), config)
    assert [(chart["left"], chart["top"]) for chart in charts[:2]] == [
        (0, 0), (550, 0),
    ]
    assert (charts[2]["left"], charts[2]["top"]) == (100, 350)
    assert all(chart["no_major_gridlines"] for chart in charts)


def test_out_of_range_space_percentage_aborts_chart_creation() -> None:
    config = _small_config()
    config["graph"]["longitudinal_profiles_percent"] = [101]
    with pytest.raises(ValueError, match="between 0 and 100"):
        build_charts(_small_result(), config)
