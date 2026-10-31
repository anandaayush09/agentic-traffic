"""Validates the post-deployment smoke test against a fake server I can make healthy, broken,
insecure, or mid-rollout. The script is the gate between "deployed" and "verified", so what
matters is that it FAILS for the situations it exists to catch -- the old version still
answering, auth accidentally off, the no-login token endpoint exposed -- not just that it
passes on a healthy service.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import smoke_test_deployment as smoke


class FakeService:
    """A programmable stand-in for the API, served over real HTTP on localhost."""

    def __init__(self) -> None:
        self.version = "sha-new"
        self.ready = True
        self.vehicles = 120
        self.auth_enforced = True
        self.demo_token_exposed = False
        self.hits: list[tuple[str, str]] = []

    def respond(self, method: str, path: str) -> tuple[int, Any]:
        self.hits.append((method, path))
        if (method, path) == ("GET", "/health"):
            return 200, {"status": "ok"}
        if (method, path) == ("GET", "/ready"):
            return (200, {"ready": True}) if self.ready else (503, {"ready": False})
        if (method, path) == ("GET", "/version"):
            return 200, {"version": self.version, "environment": "staging"}
        if (method, path) == ("GET", "/incidents"):
            return 200, []
        if (method, path) == ("GET", "/traffic"):
            return 200, {"ready": True, "stats": {"vehicles": self.vehicles}}
        if (method, path) == ("POST", "/incidents"):
            return (401, {"detail": "Missing bearer token"}) if self.auth_enforced else (201, {})
        if (method, path) == ("POST", "/auth/demo-token"):
            return (200, {"token": "x"}) if self.demo_token_exposed else (404, {"detail": "no"})
        return 404, {}


@pytest.fixture
def service() -> Iterator[tuple[FakeService, str]]:
    fake = FakeService()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _serve(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            if length:
                self.rfile.read(length)
            status, payload = fake.respond(method, self.path)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 -- http.server's required name
            self._serve("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._serve("POST")

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield fake, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _run(base: str, **kwargs: Any) -> list[smoke.CheckResult]:
    params: dict[str, Any] = {"timeout_s": 2, "interval_s": 0.2}
    params.update(kwargs)
    return smoke.run_smoke(base, **params)


def _failed(results: list[smoke.CheckResult]) -> list[str]:
    return [r.name for r in results if not r.ok]


class TestHealthyDeployment:
    def test_a_correct_deployment_passes_every_check(
        self, service: tuple[FakeService, str]
    ) -> None:
        _, base = service

        results = _run(base, expect_version="sha-new", require_live_traffic=True)

        assert _failed(results) == []
        assert len(results) == 7

    def test_live_traffic_is_only_checked_when_asked(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service

        _run(base, expect_version="sha-new")

        assert ("GET", "/traffic") not in fake.hits


class TestRollout:
    def test_waits_for_the_new_version_to_replace_the_old_one(
        self, service: tuple[FakeService, str]
    ) -> None:
        """During a rolling deploy the OLD build keeps answering for a while. A smoke test that
        passes on the first 200 would bless the old build as the new release."""
        fake, base = service
        fake.version = "sha-old"
        threading.Timer(0.6, lambda: setattr(fake, "version", "sha-new")).start()

        results = _run(base, expect_version="sha-new", timeout_s=5)

        assert _failed(results) == []

    def test_fails_if_the_old_version_never_goes_away(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service
        fake.version = "sha-old"

        results = _run(base, expect_version="sha-new", timeout_s=1)

        failure = next(r for r in results if not r.ok)
        assert "running 'sha-old', expected 'sha-new'" in failure.detail
        assert results[-1] is failure  # and it stopped there rather than probing further

    def test_waits_for_readiness(self, service: tuple[FakeService, str]) -> None:
        fake, base = service
        fake.ready = False
        threading.Timer(0.5, lambda: setattr(fake, "ready", True)).start()

        assert _failed(_run(base, expect_version="sha-new", timeout_s=5)) == []

    def test_fails_if_it_never_becomes_ready(self, service: tuple[FakeService, str]) -> None:
        fake, base = service
        fake.ready = False

        results = _run(base, expect_version="sha-new", timeout_s=1)

        assert _failed(results) == ["ready (/ready)"]

    def test_fails_if_nothing_is_listening(self) -> None:
        results = smoke.run_smoke("http://127.0.0.1:1", timeout_s=1, interval_s=0.2)

        assert not results[-1].ok
        assert "Error" in results[-1].detail or "refused" in results[-1].detail.lower()


class TestSecurityChecks:
    def test_catches_authentication_being_turned_off(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service
        fake.auth_enforced = False

        failed = _failed(_run(base, expect_version="sha-new"))

        assert any("auth enforced" in name for name in failed)

    def test_catches_the_no_login_token_endpoint_being_exposed(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service
        fake.demo_token_exposed = True

        failed = _failed(_run(base, expect_version="sha-new"))

        assert any("token endpoint hidden" in name for name in failed)

    def test_a_dead_live_simulation_fails_when_it_is_required(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service
        fake.vehicles = 0

        failed = _failed(_run(base, expect_version="sha-new", require_live_traffic=True))

        assert any("live traffic" in name for name in failed)

    def test_the_security_probe_never_creates_a_real_incident(
        self, service: tuple[FakeService, str]
    ) -> None:
        fake, base = service

        _run(base, expect_version="sha-new")

        # The only POST to /incidents is the unauthenticated probe, which a healthy API rejects.
        assert fake.hits.count(("POST", "/incidents")) == 1


class TestInputValidation:
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "example.com"])
    def test_only_http_and_https_base_urls_are_accepted(self, url: str) -> None:
        with pytest.raises(ValueError, match="must start with http"):
            smoke.run_smoke(url, timeout_s=1, interval_s=0.1)
