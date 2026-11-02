"""Simulation failures identify the calculation and river location."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transportmodel.numerical import SimulationError, run_model


ROOT = Path(__file__).resolve().parents[1]


def _heat_case(water_temperature: float, air_temperature: float) -> dict:
    config = json.loads((ROOT / "examples" / "river_example.json").read_text(encoding="utf-8"))
    config["properties"]["Generic"]["active"] = False
    temperature = config["properties"]["Temperature"]
    temperature["initial"] = {
        "type": "DEFAULT", "default_cell_value": water_temperature,
        "cells": [], "intervals": [], "interval_kind": "CELL",
        "default_cell_label_present": True,
    }
    temperature.update(
        minimum=-1000, maximum=1000, free_surface_flux=True,
        sensible_heat=False, latent_heat=True, radiative_heat=False,
    )
    config["atmosphere"]["air_temperature_c"] = air_temperature
    return config


@pytest.mark.parametrize(
    "water,air,source",
    [(-244.0, 20.0, "water"), (20.0, -244.0, "air")],
)
def test_latent_heat_overflow_names_temperature_source_and_cell(
    water: float, air: float, source: str,
) -> None:
    with pytest.raises(SimulationError) as raised:
        run_model(_heat_case(water, air))
    error = raised.value
    assert (error.stage, error.property_name, error.step,
            error.time_seconds, error.cell) == (
                "Latent heat exchange", "Temperature", 1, 60.0, 1)
    assert f"{source} temperature -244" in error.detail
    assert "exp(8622.96)" in error.detail
    assert "-244" in error.detail
    assert error.original_type == "OverflowError"
    assert error.__cause__ is not None
    assert error.hint


def test_bod_temperature_power_overflow_names_reaction_and_cell() -> None:
    fixture = json.loads((ROOT / "tests" / "fixtures" / "excel_oracle" /
                          "all_active_mixed.json").read_text(encoding="utf-8"))
    config = fixture["config"]
    temperature = config["properties"]["Temperature"]
    temperature["initial"] = {
        "type": "DEFAULT", "default_cell_value": 20000,
        "cells": [], "intervals": [], "interval_kind": "CELL",
        "default_cell_label_present": True,
    }
    temperature.update(minimum=-30000, maximum=30000, free_surface_flux=False)
    with pytest.raises(SimulationError) as raised:
        run_model(config)
    error = raised.value
    assert (error.stage, error.property_name, error.step,
            error.time_seconds, error.cell) == (
                "BOD temperature acceleration", "BOD", 1, 60.0, 1)
    assert "1.047 ** (20000 - 20)" in error.detail
    assert error.original_type == "OverflowError"


def test_invalid_input_is_reported_as_setup_failure() -> None:
    config = _heat_case(20, 20)
    config["river"]["cells"] = None
    with pytest.raises(SimulationError) as raised:
        run_model(config)
    error = raised.value
    assert error.stage == "Input setup"
    assert error.step is None and error.time_seconds is None and error.cell is None
    assert error.detail


def test_discharge_error_names_discharge_and_target_cell() -> None:
    config = _heat_case(20, 20)
    config["properties"]["Generic"]["active"] = True
    config["discharges"][0].update(cell=0, flow_rate_m3_s=0.2)
    with pytest.raises(SimulationError) as raised:
        run_model(config)
    error = raised.value
    assert error.stage == "Discharge mixing"
    assert (error.property_name, error.cell, error.step) == ("Generic", 0, 1)
    assert "Discharge 1 targets cell 0" in error.detail


def test_residence_time_error_names_problem_denominator() -> None:
    config = _heat_case(20, 20)
    config["river"].update(velocity_mode="direct", velocity_m_s=-0.000001)
    with pytest.raises(SimulationError) as raised:
        run_model(config)
    error = raised.value
    assert error.stage == "River diagnostics"
    assert error.time_seconds == 480.0
    assert "velocity + 0.000001 m/s is zero" in error.detail
