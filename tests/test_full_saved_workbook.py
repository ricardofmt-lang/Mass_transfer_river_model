"""Compare the supplied 1,000-cell, two-day Excel result in full."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from transportmodel.numerical import run_model


REFERENCE = Path(__file__).parent / "fixtures" / "saved_full_reference.json.gz"


def test_140000_saved_excel_values() -> None:
    with gzip.open(REFERENCE, "rt", encoding="utf-8") as stream:
        fixture = json.load(stream)
    result = run_model(fixture["config"])
    assert set(result.values) == {"Generic", "Temperature"}
    checked = 0
    for name, rows in fixture["outputs"].items():
        assert len(rows) == len(result.time_seconds) == 70
        for j, (excel_row, python_row) in enumerate(zip(rows, result.values[name])):
            assert excel_row[0] == result.time_seconds[j]
            assert math.isclose(excel_row[1], result.time_seconds[j] / 86400,
                                rel_tol=1e-14, abs_tol=1e-14)
            assert len(python_row) == 1000
            for i, (excel_value, python_value) in enumerate(zip(excel_row[3:], python_row), start=1):
                assert math.isclose(excel_value, python_value,
                                    rel_tol=1e-14, abs_tol=1e-12), (name, j, i)
                checked += 1
    assert checked == 140_000
