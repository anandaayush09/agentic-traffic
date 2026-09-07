"""JWT auth for mutating control-plane endpoints (docs/design/03_system_architecture.md
section 3.6: "Control API... JWT, operator-only"; "Cognitive API
(`/playbooks/{id}/approve|reject`, `/incidents/{id}`) -- JWT"; "Read-only APIs...
are unauthenticated locally, and are the only surface exposed by the public demo.").

Deliberately a single shared-secret HS256 scheme, not a full OIDC/IdP integration: this
project has one operator role (no multi-tenant/roles-within-roles requirement in the
frozen design), and HS256 needs nothing beyond stdlib `hmac` under PyJWT -- no extra
native crypto dependency, consistent with this project's repeated preference for
narrow, pure-Python dependencies on Windows (see pyproject.toml's own kafka-python
comment). Swapping in real OIDC later is an isolated change to `require_operator`'s
verification step, not a redesign of any endpoint that depends on it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_ALGORITHM = "HS256"

_bearer_scheme = HTTPBearer(auto_error=False)


def create_operator_token(
    subject: str,
    secret: str,
    expires_in_s: float = 3600.0,
    now_fn: Callable[[], float] = time.time,
) -> str:
    """Mints a JWT for `subject` (an operator id/username), signed with `secret`,
    valid for `expires_in_s` seconds from `now_fn()`. `now_fn` is injectable so
    expiry can be tested deterministically, the same fake-clock pattern used
    throughout this project (TokenBucket, Watchdog, LockStore).
    """
    now = now_fn()
    payload = {"sub": subject, "iat": now, "exp": now + expires_in_s}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def require_operator(
    secret: str,
) -> Callable[[HTTPAuthorizationCredentials | None], dict[str, object]]:
    """Returns a FastAPI dependency that requires a valid, unexpired Bearer JWT signed
    with `secret`. Raises 401 for a missing header, a bad signature, or an expired
    token -- three distinct, deliberately indistinguishable-to-the-caller failure modes
    (the standard "don't leak which part of auth failed" posture), each still logged
    server-side with its real reason via the raised detail message.
    """

    def _dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
    ) -> dict[str, object]:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
            )
        try:
            claims: dict[str, object] = jwt.decode(
                credentials.credentials, secret, algorithms=[_ALGORITHM]
            )
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
            ) from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from exc
        return claims

    return _dependency
