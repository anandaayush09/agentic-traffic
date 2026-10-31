"""Runtime configuration rules that must hold before a deployed service starts.

Local development is deliberately forgiving: a hardcoded demo JWT secret, a demo-token
button, a token printed to the console. None of that is acceptable once the service is
reachable by other people, so `ATLAS_ENV` (development | staging | production, default
development) switches on a fail-fast guard: a protected environment refuses to boot with an
unsafe configuration instead of quietly running insecurely. Refusing at startup is the point --
a misconfigured staging that serves traffic with a publicly-known signing secret is worse than
one that doesn't start.
"""

from __future__ import annotations

from collections.abc import Mapping

# The well-known value baked into local scripts, compose and the chart defaults. Anyone can read
# it in this repository, so a token signed with it proves nothing.
DEMO_JWT_SECRET = "local-demo-secret-do-not-use-in-prod-32-bytes"  # noqa: S105 -- public on purpose

PROTECTED_ENVIRONMENTS = frozenset({"staging", "production"})
MIN_JWT_SECRET_LENGTH = 32


def environment_name(env: Mapping[str, str]) -> str:
    return (env.get("ATLAS_ENV") or "development").strip().lower()


def is_protected(env: Mapping[str, str]) -> bool:
    return environment_name(env) in PROTECTED_ENVIRONMENTS


def app_version(env: Mapping[str, str]) -> str:
    """What is running: the image's git-derived tag (set at build time), else `dev`."""
    return (env.get("ATLAS_VERSION") or "dev").strip()


def validate_runtime_config(env: Mapping[str, str]) -> list[str]:
    """Problems that make `env` unsafe for its declared environment (empty = fine).
    Always empty in `development`."""
    if not is_protected(env):
        return []

    name = environment_name(env)
    problems: list[str] = []

    secret = env.get("ATLAS_JWT_SECRET", "")
    if not secret:
        problems.append(
            "ATLAS_JWT_SECRET is not set (it would fall back to the public demo secret)"
        )
    elif secret == DEMO_JWT_SECRET:
        problems.append("ATLAS_JWT_SECRET is the public demo secret from this repository")
    elif len(secret) < MIN_JWT_SECRET_LENGTH:
        problems.append(f"ATLAS_JWT_SECRET is shorter than {MIN_JWT_SECRET_LENGTH} characters")

    if env.get("ATLAS_DEMO_AUTH") == "1":
        problems.append(
            "ATLAS_DEMO_AUTH=1 hands out operator tokens without a login -- never enable it in "
            f"{name}"
        )

    if not env.get("ATLAS_DASHBOARD_ORIGINS"):
        problems.append(
            "ATLAS_DASHBOARD_ORIGINS is not set (it defaults to http://localhost:5173, which "
            f"blocks your real dashboard and is wrong for {name})"
        )
    return problems
