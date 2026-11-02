"""A failed run should explain the failed calculation and keep its traceback."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _overflow_example(tmp_path: Path) -> Path:
    """A cold input known to overflow the Excel-equivalent latent heat term."""
    payload = json.loads((ROOT / "examples" / "river_example.json").read_text(encoding="utf-8"))
    payload["properties"]["Generic"]["active"] = False
    temperature = payload["properties"]["Temperature"]
    temperature["minimum"] = -1000
    temperature["maximum"] = 1000
    temperature["initial"].update({
        "type": "DEFAULT", "default_cell_value": -244,
        "cells": [], "intervals": [], "interval_kind": "CELL",
        "default_cell_label_present": True,
    })
    temperature.update({
        "free_surface_flux": True, "sensible_heat": False,
        "latent_heat": True, "radiative_heat": False,
    })
    shutil.copy2(ROOT / "app.py", tmp_path / "app.py")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "river_example.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return tmp_path / "app.py"


def test_overflow_message_identifies_calculation_cell_and_inputs(tmp_path: Path) -> None:
    at = AppTest.from_file(str(_overflow_example(tmp_path)), default_timeout=30).run()
    at.button(key="load_example").click().run()
    at.button(key="run_model").click().run()
    assert not at.exception
    assert "result" not in at.session_state
    report = at.session_state["simulation_error_report"]
    assert "too large for the computer to represent" in report["summary"]
    assert "latent heat exchange" in report["where"].casefold()
    assert "Temperature" in report["where"]
    assert "River cell: 1" in report["where"]
    assert "-244" in report["detail"]
    assert "saturation vapour-pressure" in report["hint"]
    assert "OverflowError" in report["technical"]
    assert at.error and "Simulation stopped" in at.error[0].value
    assert at.info and "What to check" in at.info[0].value
    assert any(expander.label == "Technical details" for expander in at.expander)

    # The diagnosis remains available while the user inspects/edits inputs.
    at.run()
    assert at.session_state["simulation_error_report"] == report
    at.button(key="clear_inputs").click().run()
    assert "simulation_error_report" not in at.session_state


def test_raw_errno_overflow_is_explained_without_opaque_tuple() -> None:
    from app import _simulation_error_report

    try:
        raise OverflowError(34, "Result too large")
    except OverflowError as exc:
        report = _simulation_error_report(exc)
    assert "too large for the computer to represent" in report["summary"]
    assert "mathematical function" in report["detail"].lower()
    assert "(34, 'Result too large')" in report["technical"]


def test_live_app_rerun_refreshes_stale_solver_without_restart() -> None:
    # Streamlit can keep imported modules in memory when app.py is rerun.
    check = """
import transportmodel.numerical as numerical
import transportmodel.sources as sources
del numerical.SimulationError
del sources.CellCalculationError
import app
assert hasattr(numerical, 'SimulationError')
assert hasattr(sources, 'CellCalculationError')
assert app.run_model is numerical.run_model
"""
    completed = subprocess.run(
        [sys.executable, "-c", check], cwd=ROOT, capture_output=True,
        text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

