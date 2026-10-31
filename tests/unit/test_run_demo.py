"""Validates the one-command demo launcher's building blocks: port detection, waiting for a
server to answer (and giving up if a started process dies), and stopping a process tree.
The launcher itself starts real servers, so it is exercised by hand, not here.
"""

from __future__ import annotations

import http.server
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest

import run_demo


@pytest.fixture
def listening_port() -> Iterator[int]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.fixture
def http_server() -> Iterator[int]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- http.server's required name
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestPortInUse:
    def test_detects_a_listening_port(self, listening_port: int) -> None:
        assert run_demo.port_in_use(listening_port) is True

    def test_a_free_port_is_not_in_use(self) -> None:
        assert run_demo.port_in_use(_free_port()) is False


class TestWaitForHttp:
    def test_returns_true_once_the_server_answers(self, http_server: int) -> None:
        assert run_demo.wait_for_http(f"http://127.0.0.1:{http_server}/", 10, []) is True

    def test_gives_up_after_the_timeout(self) -> None:
        started = time.monotonic()

        ok = run_demo.wait_for_http(f"http://127.0.0.1:{_free_port()}/", 1.5, [])

        assert ok is False
        assert time.monotonic() - started < 6

    def test_gives_up_immediately_if_a_started_process_died(self, http_server: int) -> None:
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()

        ok = run_demo.wait_for_http(f"http://127.0.0.1:{http_server}/", 10, [dead])

        assert ok is False


class TestStop:
    def test_stops_a_running_process_and_its_children(self) -> None:
        # A parent that spawns a long-lived child, like `npm run dev` spawning node.
        code = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        )
        process = run_demo.start([sys.executable, "-c", code], run_demo.ROOT, {})
        time.sleep(1.0)
        assert process.poll() is None

        run_demo.stop(process)

        assert process.poll() is not None

    def test_stopping_an_already_finished_process_is_harmless(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()

        run_demo.stop(process)  # must not raise


class TestPreflight:
    def test_reports_a_busy_port(
        self, listening_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_demo, "API_PORT", listening_port)

        problems = run_demo.preflight()

        assert any(f"Port {listening_port}" in p for p in problems)


class TestNpmExecutable:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific CreateProcess behaviour")
    def test_on_windows_it_picks_the_cmd_shim_which_can_actually_be_launched(self) -> None:
        npm = run_demo.npm_executable()
        if npm is None:
            pytest.skip("npm is not installed")

        assert npm.lower().endswith(".cmd")
        # ...and it really runs (the extensionless script raised WinError 193).
        result = subprocess.run([npm, "--version"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0
