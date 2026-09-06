"""Schema-or-it-didn't-happen (docs/design/05_communication.md section 5.1, principle 4):
every bus topic has a versioned JSON Schema committed under `schemas/`, validated at
produce AND consume. `SchemaRegistry` loads these once (by topic name -> `<topic>.schema
.json`) and is the single source of truth both `BusProducer` and `BusConsumer` share, so
they can never silently drift apart on what a topic's shape is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from atlas.common.config import project_root

_DEFAULT_SCHEMAS_DIR = project_root() / "schemas"


class SchemaValidationError(ValueError):
    """A bus message did not conform to its topic's committed JSON Schema, or the topic
    has no committed schema at all."""


class SchemaRegistry:
    def __init__(self, schemas_dir: Path = _DEFAULT_SCHEMAS_DIR) -> None:
        self._schemas_dir = schemas_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def schema_for(self, topic: str) -> dict[str, Any]:
        if topic not in self._cache:
            schema_path = self._schemas_dir / f"{topic}.schema.json"
            if not schema_path.is_file():
                raise SchemaValidationError(
                    f"No committed schema for topic '{topic}' (expected {schema_path})"
                )
            self._cache[topic] = json.loads(schema_path.read_text(encoding="utf-8"))
        return self._cache[topic]

    def validate(self, topic: str, payload: dict[str, Any]) -> None:
        schema = self.schema_for(topic)
        try:
            jsonschema.validate(payload, schema)
        except jsonschema.ValidationError as exc:
            raise SchemaValidationError(
                f"Message for topic '{topic}' failed schema validation: {exc.message}"
            ) from exc
