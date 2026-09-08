#!/usr/bin/env python
"""The Week 10 roadmap's "kind smoke test (install -> scripted demo scenario ->
chaos test) in CI" (docs/design/13_roadmap.md), run for real against a real kind
cluster + the real deploy/helm/atlas chart -- not a dry-run/template-only check.

Assumes the caller has already:
    docker build -t atlas-traffic:dev .
    kind create cluster --name atlas
    kind load docker-image atlas-traffic:dev --name atlas
    helm install atlas deploy/helm/atlas -f deploy/helm/atlas/values-kind.yaml
(see the ATLAS dev-environment memory / README for the full command sequence -- kept
as explicit prerequisites here rather than shelling out to `kind`/`helm` from this
script, since orchestrating cluster lifecycle from inside the smoke test itself would
hide failures at the wrong layer: a `kind create cluster` failure and a `GET /health`
failure are different problems and should fail CI differently).

This script DOES own the `kubectl port-forward` process itself (see `PortForward`
below) -- a real finding from building this: `kubectl port-forward svc/...` tunnels to
whichever ONE pod the Service selected at forward-start time and does NOT re-resolve
to a replacement pod when that one is deleted, so the chaos test needs to kill and
restart its own forwarding process, not assume a caller-started one survives.

Usage:
    python scripts/kind_smoke_test.py --base-url http://localhost:18000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from atlas.services.auth import create_operator_token  # noqa: E402

# Must match deploy/helm/atlas/values.yaml's default cognitiveApi.jwtSecret --
# values-kind.yaml doesn't override it, so the demo/CI deployment and this script
# agree on the same JWT signing key.
JWT_SECRET = "local-demo-secret-do-not-use-in-prod-32-bytes"
SERVICE_NAME = "svc/atlas-cognitive-api"


class PortForward:
    """Owns exactly one `kubectl port-forward` subprocess for `local_port`, for this
    script's entire run -- deliberately NOT left to an externally-started process
    (see module docstring): a chaos test that deletes the backing pod needs to kill
    and relaunch this SAME subprocess afterward, and doing that reliably requires
    owning it from the start, not guessing whether an external one survived.
    """

    def __init__(self, service: str, local_port: int, remote_port: int) -> None:
        self._service = service
        self._local_port = local_port
        self._remote_port = remote_port
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            ["kubectl", "port-forward", self._service, f"{self._local_port}:{self._remote_port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3.0)  # let the tunnel actually bind before the first request

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self) -> PortForward:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


def wait_for_health(client: httpx.Client, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = client.get("/health", timeout=5.0)
            if response.status_code == 200:
                print(f"health check OK: {response.json()}")
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(
        f"Service never became healthy within {timeout_s}s (last error: {last_error})"
    )


def run_scripted_demo_scenario(client: httpx.Client, token: str) -> str:
    incident_id = f"kind-smoke-{int(time.time())}"
    response = client.post(
        "/incidents",
        json={
            "incident_id": incident_id,
            "class": "sensor_fault",
            "status": "detected",
            "confidence": 0.9,
            "location": {"link_ids": ["42430333"]},
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,  # real SUMO what-if run underneath -- give it real time
    )
    response.raise_for_status()
    body = response.json()
    assert body["state"]["proposal"]["playbook_name"] == "switch_to_max_pressure", body
    print(f"scripted demo scenario OK: incident {incident_id} -> {body['state']['status']}")
    return incident_id


def run_chaos_test(client: httpx.Client, port_forward: PortForward, timeout_s: float) -> None:
    """Kills the running pod and verifies the Deployment's own self-healing brings a
    replacement pod back to Ready, and the service is reachable again -- the literal
    "kill an agent, drop a partition... published p50/p99 latency" operational-honesty
    claim (docs/design/00_executive_summary.md), applied to k8s's own restart
    guarantee rather than assumed to just work.
    """
    pods = (
        subprocess.run(
            ["kubectl", "get", "pods", "-l", "app.kubernetes.io/name=cognitive-api", "-o", "name"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if not pods:
        raise RuntimeError("No cognitive-api pod found to kill for the chaos test")
    pod_name = pods[0]
    subprocess.run(["kubectl", "delete", pod_name], check=True)
    print(f"chaos test: deleted {pod_name}, waiting for a replacement to become Ready")
    time.sleep(2.0)  # give the Deployment controller a moment to schedule a
    # replacement pod before `kubectl wait` runs -- otherwise it can race against a
    # selector that briefly matches zero pods.
    subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            "app.kubernetes.io/name=cognitive-api",
            "--timeout",
            f"{int(timeout_s)}s",
        ],
        check=True,
    )

    port_forward.restart()
    wait_for_health(client, timeout_s)
    print("chaos test OK: service recovered after pod deletion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:18000")
    parser.add_argument("--health-timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    parsed = urlparse(args.base_url)
    local_port = parsed.port or 80

    token = create_operator_token("kind-smoke-test", JWT_SECRET)
    with (
        PortForward(SERVICE_NAME, local_port, 8000) as port_forward,
        httpx.Client(base_url=args.base_url) as client,
    ):
        wait_for_health(client, args.health_timeout_s)
        run_scripted_demo_scenario(client, token)
        run_chaos_test(client, port_forward, args.health_timeout_s)

    print("\n=== kind smoke test: ALL PASSED ===")


if __name__ == "__main__":
    main()
