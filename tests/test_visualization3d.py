"""Verify that the animated river uses actual model cells and printed times."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from transportmodel.numerical import SimulationResult
from transportmodel.visualization3d import build_river_3d_figure


def _result() -> SimulationResult:
    return SimulationResult(
        time_seconds=[0.0, 60.0, 120.0],
        coordinates_m=[5.0, 15.0, 25.0],
        values={"DO": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 5.0, 4.0]],
                "IntegratedAeroResp(kg)": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0],
                                           [2.0, 4.0, 6.0]]},
    )


@pytest.mark.parametrize("name,units", [("DO", "mg/L"), ("IntegratedAeroResp(kg)", "kg")])
def test_each_frame_is_the_selected_property_at_its_printed_time(name: str, units: str) -> None:
    result = _result()
    fig = build_river_3d_figure(result, name, width_m=4, depth_m=2,
                                length_m=30, units=units)
    assert len(fig.frames) == len(result.time_seconds) == 3
    assert len(fig.layout.sliders[0].steps) == 3
    assert len(fig.data) == 6  # top, both banks, both ends, and river bed
    assert fig.data[0].colorbar.title.text == f"{name} ({units})"
    assert list(fig.data[0].x[0]) == [0, 10, 10, 20, 20, 30]
    assert list(fig.data[0].y[0]) == [0] * 6
    assert list(fig.data[0].y[1]) == [4] * 6
    assert list(fig.data[0].z[0]) == [2] * 6
    assert list(fig.layout.scene.yaxis.range) == [0, 4]
    assert list(fig.layout.scene.zaxis.range) == [0, 2]
    for bank, across in ((fig.data[1], 0), (fig.data[2], 4)):
        assert list(bank.x[0]) == list(fig.data[0].x[0])
        assert list(bank.y[0]) == [across] * 6
        assert list(bank.y[1]) == [across] * 6
        assert list(bank.z[0]) == [0] * 6
        assert list(bank.z[1]) == [2] * 6
    for end, distance in ((fig.data[3], 0), (fig.data[4], 30)):
        assert list(end.x[0]) == [distance, distance]
        assert list(end.y[0]) == [0, 4]
        assert list(end.z[0]) == [0, 0]
        assert list(end.z[1]) == [2, 2]
    assert list(fig.data[5].z[0]) == [0] * 6
    assert list(fig.data[5].y[1]) == [4] * 6
    assert fig.layout.scene.aspectratio.x > fig.layout.scene.aspectratio.y
    for index, frame in enumerate(fig.frames):
        expected = [value for cell in result.values[name][index] for value in (cell, cell)]
        assert list(frame.traces) == [0, 1, 2, 3, 4, 5]
        for face in (*frame.data[:3], frame.data[5]):
            assert list(face.surfacecolor[0]) == expected
            assert list(face.surfacecolor[1]) == expected
        assert list(frame.data[3].surfacecolor[0]) == [expected[0]] * 2
        assert list(frame.data[4].surfacecolor[0]) == [expected[-1]] * 2
        assert f"{result.time_seconds[index]:g} s" in frame.layout.title.text
    assert fig.data[0].cmin == min(value for row in result.values[name] for value in row)
    assert fig.data[0].cmax == max(value for row in result.values[name] for value in row)
    assert '"frames"' in fig.to_json()


def test_entered_depth_changes_channel_geometry_not_property_distribution() -> None:
    result = _result()
    shallow = build_river_3d_figure(result, "DO", width_m=4, depth_m=1,
                                    length_m=30)
    deep = build_river_3d_figure(result, "DO", width_m=4, depth_m=3,
                                 length_m=30)
    assert list(shallow.layout.scene.zaxis.range) == [0, 1]
    assert list(deep.layout.scene.zaxis.range) == [0, 3]
    assert list(shallow.data[1].z[1]) == [1] * 6
    assert list(deep.data[1].z[1]) == [3] * 6
    assert shallow.layout.scene.aspectratio.z < deep.layout.scene.aspectratio.z
    for face in range(6):
        assert shallow.data[face].surfacecolor == deep.data[face].surfacecolor


def test_constant_property_still_has_a_visible_color_scale() -> None:
    result = SimulationResult(time_seconds=[0], coordinates_m=[5], values={"CO2": [[1.5]]})
    fig = build_river_3d_figure(result, "CO2", width_m=2, depth_m=1, length_m=10)
    assert fig.data[0].cmin < 1.5 < fig.data[0].cmax
    assert len(fig.frames) == 1


def test_streamlit_result_includes_three_dimensional_animation() -> None:
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    at.button(key="load_example").click().run()
    at.button(key="run_model").click().run()
    assert not at.exception
    assert "3D river animation" in [tab.label for tab in at.tabs]
    # Three ordinary plots, one 3D river, and one workbook-style plot.
    assert len(at.get("plotly_chart")) == 5
