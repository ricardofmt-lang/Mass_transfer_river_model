"""The installed Python app runs without the source XLSM or Excel process."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_and_charts_run_with_only_python_and_bundled_json(tmp_path: Path) -> None:
    shutil.copy2(ROOT / "app.py", tmp_path / "app.py")
    shutil.copytree(
        ROOT / "transportmodel", tmp_path / "transportmodel",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (tmp_path / "examples").mkdir()
    shutil.copy2(ROOT / "examples" / "river_example.json",
                 tmp_path / "examples" / "river_example.json")
    assert not list(tmp_path.rglob("*.xls*"))

    check = textwrap.dedent("""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file('app.py', default_timeout=30).run()
        assert not app.exception, app.exception
        app.button(key='load_example').click().run()
        assert not app.exception, app.exception
        app.button(key='run_model').click().run()
        assert not app.exception, app.exception
        assert {'Generic', 'Temperature'} <= set(app.session_state['result'].values)
        assert len(app.get('plotly_chart')) == 5
    """)
    completed = subprocess.run(
        [sys.executable, "-c", check], cwd=tmp_path, capture_output=True,
        text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
