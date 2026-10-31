#!/usr/bin/env python
"""Post-deployment smoke test: is the thing that was just deployed actually serving, and is it
the build we meant to ship?

    python scripts/smoke_test_deployment.py --base-url https://atlas-staging.example.com \\
        --expect-version sha-abc1234 [--require-live-traffic] [--timeout 300]

Standard library only (it runs on a bare CI runner, no install step) and read-only apart from
one deliberately-unauthenticated POST that must be REJECTED.

What it checks, and why each one earns its place:

1. `/health` is 200       -- the process is up.
2. `/ready` is 200        -- it is fit to take traffic (the live simulation finished warming up).
3. `/version` == expected -- the NEW build is the one answering. During a rolling deploy the old
                             instance keeps serving for a while, so a bare "it returned 200"
                             can be the OLD version passing a check meant for the new one. The
                             test keeps polling until the expected version shows up or time runs out.
4. `GET /incidents` is a JSON list -- the API's core read path works end to end.
5. `GET /traffic` has vehicles (with --require-live-traffic) -- the simulation is really running.
6. Unauthenticated `POST /incidents` is 401 -- authentication is actually enforced. A deployment
                             that accidentally turned auth off is "healthy" by every other
                             check, which is exactly why this one exists.
7. `POST /auth/demo-token` is 404 -- the no-login token endpoint is not exposed.

Checks 1-3 are retried until `--timeout` (a deploy takes time to roll out); 4-7 run once the
service is ready. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _request(
    base_url: str, path: str, method: str = "GET", body: bytes | None = None, timeout_s: float = 10
) -> tuple[int, str]:
    """(status, body). An HTTP error status is a result, not an exception; only a connection
    failure raises (as OSError)."""
    # The base URL comes from a CI variable: accept only http(s) so a mistyped or hostile value
    # can't make urllib open a file:// or ftp:// URL.
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"--base-url must start with http:// or https://, got {base_url!r}")
    request = urllib.request.Request(  # noqa: S310 -- scheme validated above
        base_url.rstrip("/") + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _json(body: str) -> object:
    try:
        return json.loads(body)
    except ValueError:
        return None


def _poll(check: Callable[[], CheckResult], timeout_s: float, interval_s: float) -> CheckResult:
    """Re-runs `check` until it passes or `timeout_s` elapses; returns the last result."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            result = check()
        except OSError as exc:  # connection refused/timeout: the service isn't up yet
            result = CheckResult("connect", False, f"{type(exc).__name__}: {exc}")
        if result.ok or time.monotonic() >= deadline:
            return result
        time.sleep(interval_s)


def run_smoke(
    base_url: str,
    expect_version: str | None = None,
    require_live_traffic: bool = False,
    timeout_s: float = 300,
    interval_s: float = 5,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    def health() -> CheckResult:
        status, body = _request(base_url, "/health")
        ok = status == 200 and _json(body) == {"status": "ok"}
        return CheckResult("health (/health)", ok, f"HTTP {status}")

    def ready() -> CheckResult:
        status, body = _request(base_url, "/ready")
        return CheckResult("ready (/ready)", status == 200, f"HTTP {status} {body[:120]}")

    def version() -> CheckResult:
        status, body = _request(base_url, "/version")
        data = _json(body)
        running = data.get("version") if isinstance(data, dict) else None
        if status != 200:
            return CheckResult("version (/version)", False, f"HTTP {status}")
        if expect_version is not None and running != expect_version:
            return CheckResult(
                "version (/version)", False, f"running {running!r}, expected {expect_version!r}"
            )
        return CheckResult("version (/version)", True, f"running {running!r}")

    for check in (health, ready, version):
        result = _poll(check, timeout_s, interval_s)
        results.append(result)
        if not result.ok:
            return results  # no point probing a service that isn't (the right) one

    def incidents() -> CheckResult:
        status, body = _request(base_url, "/incidents")
        ok = status == 200 and isinstance(_json(body), list)
        return CheckResult("incidents list (GET /incidents)", ok, f"HTTP {status}")

    def traffic() -> CheckResult:
        status, body = _request(base_url, "/traffic")
        data = _json(body)
        stats = data.get("stats") if isinstance(data, dict) else None
        vehicles = stats.get("vehicles", 0) if isinstance(stats, dict) else 0
        ok = status == 200 and isinstance(data, dict) and data.get("ready") is True and vehicles > 0
        return CheckResult("live traffic (GET /traffic)", ok, f"HTTP {status}, {vehicles} vehicles")

    def auth_enforced() -> CheckResult:
        probe = json.dumps({"incident_id": "smoke-probe", "class": "unknown"}).encode()
        status, _ = _request(base_url, "/incidents", method="POST", body=probe)
        return CheckResult(
            "auth enforced (unauthenticated POST /incidents -> 401)",
            status == 401,
            f"HTTP {status}" + ("" if status == 401 else "  <-- AUTH MAY BE OFF"),
        )

    def demo_token_hidden() -> CheckResult:
        status, _ = _request(base_url, "/auth/demo-token", method="POST", body=b"")
        return CheckResult(
            "no-login token endpoint hidden (POST /auth/demo-token -> 404)",
            status == 404,
            f"HTTP {status}" + ("" if status == 404 else "  <-- DEMO AUTH MAY BE ON"),
        )

    checks: list[Callable[[], CheckResult]] = [incidents, auth_enforced, demo_token_hidden]
    if require_live_traffic:
        checks.insert(1, traffic)
    for check in checks:
        try:
            results.append(check())
        except OSError as exc:
            results.append(CheckResult("request", False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expect-version", help="the image tag that must be the one answering")
    parser.add_argument("--require-live-traffic", action="store_true")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for rollout")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()

    print(f"Smoke test: {args.base_url} (expecting version {args.expect_version or 'any'})")
    results = run_smoke(
        args.base_url,
        args.expect_version,
        args.require_live_traffic,
        args.timeout,
        args.interval,
    )
    for result in results:
        print(f"  [{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
    failed = [r for r in results if not r.ok]
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
