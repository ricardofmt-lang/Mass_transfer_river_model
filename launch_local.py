"""Start the local app on an available port, or reuse its running instance.

Called by run.cmd so PowerShell script execution policy is irrelevant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
from typing import Any, Callable
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
FIRST_PORT = 8502
LAST_PORT = 8599


# Ask Windows for structured process data so port ownership can be checked reliably.
def _powershell_json(command: str) -> Any:
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=12, check=False,
        )
        return json.loads(completed.stdout) if completed.returncode == 0 and completed.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


# Map listening TCP ports to the process IDs that own them.
def _listener_owners() -> dict[int, list[int]]:
    command = (
        "Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.LocalPort -ge {FIRST_PORT} -and $_.LocalPort -le {LAST_PORT} }} | "
        "Select-Object LocalPort,OwningProcess | ConvertTo-Json -Compress"
    )
    data = _powershell_json(command)
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    owners: dict[int, list[int]] = {}
    for item in items:
        try:
            owners.setdefault(int(item["LocalPort"]), []).append(int(item["OwningProcess"]))
        except (KeyError, TypeError, ValueError):
            continue
    return owners


# Inspect a candidate process before assuming it is this river app.
def _process(pid: int) -> dict[str, Any] | None:
    data = _powershell_json(
        f'Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" | '
        'Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | '
        'ConvertTo-Json -Compress'
    )
    return data if isinstance(data, dict) else None


# Reuse a port only if its owner is this app and its health endpoint responds.
def _same_app(pid: int, port: int, process: Callable[[int], dict[str, Any] | None] = _process) -> bool:
    """Match the Streamlit command and this project's venv in its ancestry."""
    seen: set[int] = set()
    streamlit_command = False
    project_python = os.path.normcase(os.path.abspath(PYTHON))
    for _ in range(8):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        info = process(pid)
        if info is None:
            break
        command = str(info.get("CommandLine") or "")
        if (re.search(r"-m\s+streamlit\s+run\s+(?:\S+[\\/])?app\.py(?:\s|$)", command, re.I)
                and re.search(rf"--server\.port\s+{port}(?:\s|$)", command)):
            streamlit_command = True
        executable = str(info.get("ExecutablePath") or "")
        if streamlit_command and os.path.normcase(os.path.abspath(executable)) == project_python:
            return True
        try:
            pid = int(info.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            break
    return False


# A Streamlit health response confirms the existing local server is ready.
def _healthy(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except OSError:
        return False


# Bind to loopback briefly to find an unused port without exposing the service remotely.
def _free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


# Prefer the customary river port, then choose the next free local port.
def choose_port(
    owners: dict[int, list[int]],
    same_app: Callable[[int, int], bool],
    healthy: Callable[[int], bool],
    free: Callable[[int], bool],
    first: int = FIRST_PORT,
    last: int = LAST_PORT,
) -> tuple[str, int]:
    for port in range(first, last + 1):
        if any(same_app(pid, port) for pid in owners.get(port, [])) and healthy(port):
            return "existing", port
    for port in range(first, last + 1):
        if free(port):
            return "new", port
    raise RuntimeError(f"No free local port from {first} through {last}")


# Start the app with the project interpreter and print the URL to open.
def main() -> int:
    if not PYTHON.is_file():
        print("Create the environment with: py -3.12 -m venv .venv")
        print("Then install: .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return 1
    try:
        action, port = choose_port(_listener_owners(), _same_app, _healthy, _free)
    except RuntimeError as exc:
        print(exc)
        return 1
    url = f"http://127.0.0.1:{port}/"
    if action == "existing":
        print(f"River app is already running at {url}")
        return 0
    if port != FIRST_PORT:
        print(f"Port {FIRST_PORT} is occupied; using port {port}.", flush=True)
    print(f"Open the river app at {url}", flush=True)
    try:
        return subprocess.call([
            str(PYTHON), "-m", "streamlit", "run", "app.py",
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
        ], cwd=ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
