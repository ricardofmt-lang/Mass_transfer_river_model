"""Port conflicts must not make a second local run fail or show another app."""

from __future__ import annotations

from launch_local import PYTHON, _same_app, choose_port


def test_reuses_this_river_app_when_it_owns_8502() -> None:
    assert choose_port(
        {8502: [17]}, lambda pid, port: (pid, port) == (17, 8502),
        lambda port: port == 8502, lambda port: False, first=8502, last=8503,
    ) == ("existing", 8502)


def test_another_app_on_8502_moves_river_to_next_free_port() -> None:
    assert choose_port(
        {8502: [19]}, lambda pid, port: False, lambda port: False,
        lambda port: port == 8503, first=8502, last=8504,
    ) == ("new", 8503)


def test_recognizes_venv_ancestor_of_streamlit_listener() -> None:
    processes = {
        21: {"ParentProcessId": 20, "ExecutablePath": r"C:\Python\python.exe",
             "CommandLine": r"python -m streamlit run app.py --server.port 8502"},
        20: {"ParentProcessId": 0, "ExecutablePath": str(PYTHON),
             "CommandLine": r"python -m streamlit run app.py --server.port 8502"},
    }
    assert _same_app(21, 8502, processes.get)
    assert not _same_app(21, 8503, processes.get)
    processes[20]["ExecutablePath"] = r"C:\OtherApp\.venv\Scripts\python.exe"
    assert not _same_app(21, 8502, processes.get)
