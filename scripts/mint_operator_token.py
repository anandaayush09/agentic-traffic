#!/usr/bin/env python
"""Mints a short-lived operator JWT for a deployed API, from the SAME secret the API uses.

The API no longer prints a token at startup outside local development (a token in a container's
logs would hand operator rights to anyone who can read logs). To approve incidents on staging or
production, mint one deliberately, with the environment's own secret, for as short a time as you need:

    ATLAS_JWT_SECRET=... python scripts/mint_operator_token.py --subject alice --hours 1

The secret is read from the environment only (never an argument, which shell history and process
lists would record), and the token is printed to stdout so it can be piped or copied.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.runtime_config import DEMO_JWT_SECRET  # noqa: E402
from atlas.services.auth import create_operator_token  # noqa: E402


def mint(subject: str, hours: float, secret: str) -> str:
    if not secret:
        raise ValueError("ATLAS_JWT_SECRET is not set")
    if secret == DEMO_JWT_SECRET:
        raise ValueError(
            "ATLAS_JWT_SECRET is the public demo secret -- a token signed with it is meaningless "
            "on a protected deployment (and the API refuses to start with it there)"
        )
    if not 0 < hours <= 24:
        raise ValueError("--hours must be between 0 and 24")
    return create_operator_token(subject, secret, expires_in_s=hours * 3600.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subject", required=True, help="who this token is for (audit trail)")
    parser.add_argument("--hours", type=float, default=1.0, help="lifetime (default 1h, max 24h)")
    args = parser.parse_args()
    try:
        print(mint(args.subject, args.hours, os.environ.get("ATLAS_JWT_SECRET", "")))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
