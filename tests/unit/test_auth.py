"""Validates the JWT dependency against a real FastAPI app + real PyJWT encode/decode
round trip -- no mocking of jwt itself, since a bug in signature/expiry handling here
is exactly the kind of thing worth catching for real.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from atlas.services.auth import create_operator_token, require_operator

SECRET = "test-secret-do-not-use-in-prod-32-bytes-min"


def _protected_app(secret: str = SECRET) -> FastAPI:
    app = FastAPI()
    require_jwt = require_operator(secret)

    @app.get("/protected")
    def protected(claims: dict[str, object] = Depends(require_jwt)) -> dict[str, object]:  # noqa: B008
        return {"ok": True, "sub": claims["sub"]}

    return app


class TestMissingOrMalformedToken:
    def test_no_authorization_header_is_rejected(self) -> None:
        client = TestClient(_protected_app())

        response = client.get("/protected")

        assert response.status_code == 401

    def test_garbage_token_is_rejected(self) -> None:
        client = TestClient(_protected_app())

        response = client.get("/protected", headers={"Authorization": "Bearer not-a-real-jwt"})

        assert response.status_code == 401


class TestValidToken:
    def test_correctly_signed_unexpired_token_is_accepted(self) -> None:
        token = create_operator_token("operator-1", SECRET, expires_in_s=3600.0)
        client = TestClient(_protected_app())

        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        assert response.json() == {"ok": True, "sub": "operator-1"}


class TestWrongSecret:
    def test_token_signed_with_a_different_secret_is_rejected(self) -> None:
        token = create_operator_token("operator-1", "a-completely-different-secret-32-bytes-min")
        client = TestClient(_protected_app())

        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401


class TestExpiry:
    def test_expired_token_is_rejected(self) -> None:
        """now_fn is faked to mint a token whose exp is already in the past --
        deterministic, no real sleeping needed to prove expiry actually works."""
        fake_now = 1_000_000.0
        token = create_operator_token(
            "operator-1", SECRET, expires_in_s=60.0, now_fn=lambda: fake_now - 3600.0
        )
        client = TestClient(_protected_app())

        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401
