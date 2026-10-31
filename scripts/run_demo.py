#!/usr/bin/env python
"""One command to run the whole demo: the API (live traffic, demo-token button enabled)
and the dashboard dev server, together, stopped together.

    python scripts/run_demo.py            # or: make demo

It waits until both answer, prints the dashboard URL, and on Ctrl+C stops everything it
started (including the child processes npm spawns). Preconditions it checks and explains
rather than failing mysteriously: the road network is built, `npm install` has been run,
and ports 8000 / 5173 are free (the API only allows the dashboard's own origin,
http://localhost:5173, so the dashboard must get exactly that port -- hence `--strictPort`).

Keys for LLM proposals (GROQ_API_KEY / GEMINI_API_KEY) are read from `.env` by the API as
usual; set ATLAS_STATE_DB=<file> to keep incident history across runs. The demo-token
button is enabled only here (ATLAS_DEMO_AUTH=1), never by the API script on its own.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
API_PORT = 8000
DASHBOARD_PORT = 5173
NET_FILE = ROOT / "data" / "networks" / "manhattan_midtown" / "net.net.xml"


def npm_executable() -> str | None:
    """Path to something runnable as `npm`. On Windows `shutil.which("npm")` finds the
    extensionless shell script Node installs next to `npm.cmd`, and CreateProcess cannot run
    that (WinError 193) -- the `.cmd` shim is the one that works."""
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm")
    return shutil.which("npm")


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_http(url: str, timeout_s: float, processes: list[subprocess.Popen[bytes]]) -> bool:
    """True once `url` answers 200; False on timeout or if a started process died first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            return False
        try:
            # `url` is always a http://localhost URL built by this script, never user input.
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def start(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {"cwd": str(cwd), "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[call-overload]


def stop(process: subprocess.Popen[bytes]) -> None:
    """Stops `process` and everything it spawned (npm starts node; the API starts SUMO)."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def preflight() -> list[str]:
    problems: list[str] = []
    if not NET_FILE.is_file():
        problems.append(
            "The road network isn't built. Run:\n"
            "    python scripts/build_district_network.py\n"
            "    python scripts/generate_demand.py"
        )
    if not (FRONTEND / "node_modules").is_dir():
        problems.append("Dashboard dependencies missing. Run:\n    cd frontend && npm install")
    if npm_executable() is None:
        problems.append("`npm` was not found on PATH (install Node.js).")
    for port, what in ((API_PORT, "API"), (DASHBOARD_PORT, "dashboard")):
        if port_in_use(port):
            problems.append(
                f"Port {port} ({what}) is already in use -- stop whatever is using it first."
            )
    return problems


def main() -> int:
    problems = preflight()
    if problems:
        print("Can't start the demo yet:\n")
        print("\n\n".join(f"  - {p}" for p in problems))
        return 1

    api_env = {**os.environ, "ATLAS_DEMO_AUTH": "1", "PYTHONUNBUFFERED": "1"}
    started: list[subprocess.Popen[bytes]] = []
    try:
        print("Starting the API (live traffic simulation warms up for a few seconds)...")
        api = start(
            [sys.executable, str(ROOT / "scripts" / "run_dashboard_demo_api.py")], ROOT, api_env
        )
        started.append(api)
        if not wait_for_http(f"http://localhost:{API_PORT}/health", 90, started):
            print("\nThe API did not come up (see its output above).")
            return 1

        print("Starting the dashboard...")
        npm = npm_executable() or "npm"
        dashboard = start(
            [npm, "run", "dev", "--", "--port", str(DASHBOARD_PORT), "--strictPort"],
            FRONTEND,
            dict(os.environ),
        )
        started.append(dashboard)
        if not wait_for_http(f"http://localhost:{DASHBOARD_PORT}/", 60, started):
            print("\nThe dashboard did not come up (see its output above).")
            return 1

        print(
            f"\n  ATLAS is running:  http://localhost:{DASHBOARD_PORT}\n"
            "  Click 'Get demo token' in the dashboard, then try 'Demo: accident'.\n"
            "  Press Ctrl+C here to stop everything.\n"
        )
        while all(p.poll() is None for p in started):
            time.sleep(1.0)
        print("A process exited -- shutting down.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        for process in reversed(started):
            stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
