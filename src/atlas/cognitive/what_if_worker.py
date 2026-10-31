"""Runs ONE what-if in its own process (see `atlas.cognitive.what_if._run_on_tile`).

    python -m atlas.cognitive.what_if_worker request.json

The request names a (tile) network, its routes and the proposal; the result is printed as a single
JSON line. libsumo is imported first: on Windows it must be loaded before sqlite3/pyarrow/pyproj.
"""

from __future__ import annotations

import json  # noqa: E402,I001
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import libsumo  # noqa: F401  -- must precede the imports below (Windows DLL order)

from atlas.cognitive.playbooks import PlaybookProposal  # noqa: E402
from atlas.cognitive.what_if import result_as_json, run_what_if  # noqa: E402


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run_what_if(
        PlaybookProposal.model_validate(request["proposal"]),
        Path(request["net_file"]),
        Path(request["route_file"]),
        seed=int(request["seed"]),
        additional_files=[Path(p) for p in request["additional_files"]],
        incident=request["incident"],
        whole_network=True,
    )
    print(result_as_json(result), flush=True)


if __name__ == "__main__":
    main()
