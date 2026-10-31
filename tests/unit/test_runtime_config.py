"""Validates the fail-fast guard for protected environments, Sentry's opt-in initialisation and
the operator-token minting script. These are the controls that stop a staging/production
deploy from silently running with the public demo secret, so each rule is tested by the
unsafe input it exists to reject, not just the happy path.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import mint_operator_token
from atlas.common.observability import init_sentry
from atlas.common.runtime_config import (
    DEMO_JWT_SECRET,
    app_version,
    environment_name,
    is_protected,
    validate_runtime_config,
)
from atlas.services.auth import create_operator_token, require_operator

STRONG_SECRET = "x" * 48
GOOD_PRODUCTION_ENV = {
    "ATLAS_ENV": "production",
    "ATLAS_JWT_SECRET": STRONG_SECRET,
    "ATLAS_DASHBOARD_ORIGINS": "https://atlas.example.com",
}


class TestEnvironmentName:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({}, "development"),
            ({"ATLAS_ENV": ""}, "development"),
            ({"ATLAS_ENV": " Production "}, "production"),
            ({"ATLAS_ENV": "staging"}, "staging"),
        ],
    )
    def test_defaults_to_development_and_normalises(
        self, env: dict[str, str], expected: str
    ) -> None:
        assert environment_name(env) == expected

    def test_only_staging_and_production_are_protected(self) -> None:
        assert is_protected({"ATLAS_ENV": "staging"})
        assert is_protected({"ATLAS_ENV": "production"})
        assert not is_protected({})
        assert not is_protected({"ATLAS_ENV": "development"})
        assert not is_protected({"ATLAS_ENV": "prod-ish"})  # unknown names get no free pass...

    def test_version_defaults_to_dev(self) -> None:
        assert app_version({}) == "dev"
        assert app_version({"ATLAS_VERSION": "sha-abc1234"}) == "sha-abc1234"


class TestValidateRuntimeConfig:
    def test_development_accepts_anything_including_the_demo_secret(self) -> None:
        assert validate_runtime_config({}) == []
        assert validate_runtime_config({"ATLAS_JWT_SECRET": DEMO_JWT_SECRET}) == []
        assert validate_runtime_config({"ATLAS_ENV": "development", "ATLAS_DEMO_AUTH": "1"}) == []

    @pytest.mark.parametrize("env_name", ["staging", "production"])
    def test_a_properly_configured_protected_environment_passes(self, env_name: str) -> None:
        assert validate_runtime_config({**GOOD_PRODUCTION_ENV, "ATLAS_ENV": env_name}) == []

    def test_a_missing_secret_is_rejected(self) -> None:
        env = {k: v for k, v in GOOD_PRODUCTION_ENV.items() if k != "ATLAS_JWT_SECRET"}

        problems = validate_runtime_config(env)

        assert any("ATLAS_JWT_SECRET is not set" in p for p in problems)

    def test_the_public_demo_secret_is_rejected(self) -> None:
        problems = validate_runtime_config(
            {**GOOD_PRODUCTION_ENV, "ATLAS_JWT_SECRET": DEMO_JWT_SECRET}
        )

        assert any("public demo secret" in p for p in problems)

    def test_a_short_secret_is_rejected(self) -> None:
        problems = validate_runtime_config({**GOOD_PRODUCTION_ENV, "ATLAS_JWT_SECRET": "short"})

        assert any("shorter than 32" in p for p in problems)

    @pytest.mark.parametrize("env_name", ["staging", "production"])
    def test_the_demo_token_endpoint_is_forbidden_in_protected_environments(
        self, env_name: str
    ) -> None:
        problems = validate_runtime_config(
            {**GOOD_PRODUCTION_ENV, "ATLAS_ENV": env_name, "ATLAS_DEMO_AUTH": "1"}
        )

        assert any("ATLAS_DEMO_AUTH" in p for p in problems)

    def test_unset_cors_origins_are_rejected(self) -> None:
        env = {k: v for k, v in GOOD_PRODUCTION_ENV.items() if k != "ATLAS_DASHBOARD_ORIGINS"}

        assert any("ATLAS_DASHBOARD_ORIGINS" in p for p in validate_runtime_config(env))

    def test_every_problem_is_reported_at_once(self) -> None:
        problems = validate_runtime_config({"ATLAS_ENV": "production", "ATLAS_DEMO_AUTH": "1"})

        assert len(problems) == 3  # secret, demo auth, origins -- fix them in one go


class TestInitSentry:
    def test_without_a_dsn_it_is_off_and_never_imports_the_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # an import would now raise

        assert init_sentry({}, release="r", environment="e") is False
        assert init_sentry({"SENTRY_DSN": "  "}, release="r", environment="e") is False

    def test_with_a_dsn_it_initialises_conservatively(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        fake = types.ModuleType("sentry_sdk")
        fake.init = lambda **kwargs: captured.update(kwargs)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

        enabled = init_sentry(
            {"SENTRY_DSN": "https://key@example.ingest.sentry.io/1"},
            release="sha-abc1234",
            environment="staging",
        )

        assert enabled is True
        assert captured["release"] == "sha-abc1234"
        assert captured["environment"] == "staging"
        assert captured["send_default_pii"] is False
        assert captured["max_request_body_size"] == "never"  # tokens/incident payloads stay home
        assert captured["traces_sample_rate"] == 0.0

    def test_a_trace_sample_rate_can_be_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        fake = types.ModuleType("sentry_sdk")
        fake.init = lambda **kwargs: captured.update(kwargs)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

        init_sentry(
            {"SENTRY_DSN": "https://k@e.io/1", "SENTRY_TRACES_SAMPLE_RATE": "0.25"}, "r", "e"
        )

        assert captured["traces_sample_rate"] == 0.25

    def test_a_missing_sdk_degrades_to_off_instead_of_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentry_sdk", None)

        assert init_sentry({"SENTRY_DSN": "https://k@e.io/1"}, "r", "e") is False

    def test_an_sdk_that_rejects_the_dsn_never_stops_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_: Any) -> None:
            raise ValueError("Unsupported scheme")

        fake = types.ModuleType("sentry_sdk")
        fake.init = boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

        assert init_sentry({"SENTRY_DSN": "not-a-dsn"}, "r", "e") is False


class TestMintOperatorToken:
    def test_the_token_verifies_against_the_same_secret_the_api_uses(self) -> None:
        token = mint_operator_token.mint("alice", 1.0, STRONG_SECRET)

        claims = require_operator(STRONG_SECRET)(
            credentials=types.SimpleNamespace(credentials=token)  # type: ignore[arg-type]
        )
        assert claims["sub"] == "alice"

    def test_the_token_expires_when_asked(self) -> None:
        token = mint_operator_token.mint("alice", 0.5, STRONG_SECRET)
        import jwt

        claims = jwt.decode(token, STRONG_SECRET, algorithms=["HS256"])

        assert claims["exp"] - claims["iat"] == pytest.approx(1800)

    def test_the_public_demo_secret_is_refused(self) -> None:
        with pytest.raises(ValueError, match="public demo secret"):
            mint_operator_token.mint("alice", 1.0, DEMO_JWT_SECRET)

    def test_a_missing_secret_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not set"):
            mint_operator_token.mint("alice", 1.0, "")

    @pytest.mark.parametrize("hours", [0, -1, 25])
    def test_lifetimes_outside_zero_to_24_hours_are_refused(self, hours: float) -> None:
        with pytest.raises(ValueError, match="between 0 and 24"):
            mint_operator_token.mint("alice", hours, STRONG_SECRET)

    def test_a_token_signed_with_another_secret_is_rejected_by_the_api(self) -> None:
        token = create_operator_token("mallory", "a-different-secret-entirely-xxxxxxxxxxxx")

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as info:
            require_operator(STRONG_SECRET)(
                credentials=types.SimpleNamespace(credentials=token)  # type: ignore[arg-type]
            )
        assert info.value.status_code == 401
