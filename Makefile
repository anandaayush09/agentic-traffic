# ATLAS developer entrypoints. `make help` lists every target.
#
# Week 1 status: `up` is a real but currently-empty compose stack (see docker-compose.yml
# for why). `train`, `eval`, and `demo` are not implemented yet -- they're planned Week 5+
# targets and are listed here so the intended developer surface is documented, but they
# fail loudly rather than silently doing nothing.

.PHONY: help up down test smoke lint fmt typecheck train eval demo clean

help:
	@echo "ATLAS developer commands:"
	@echo "  make up          - start the local compose stack (docker compose up -d)"
	@echo "  make down        - stop the local compose stack"
	@echo "  make test        - run the test suite (pytest)"
	@echo "  make smoke       - run the digital-twin smoke test (1000 sim steps)"
	@echo "  make lint        - run ruff (lint + format check)"
	@echo "  make fmt         - auto-format with ruff"
	@echo "  make typecheck   - run mypy in strict mode"
	@echo "  make clean       - remove caches and generated sandbox data"
	@echo "  make train       - [not yet implemented, planned Week 5+]"
	@echo "  make eval        - [not yet implemented, planned Week 11]"
	@echo "  make demo        - [not yet implemented, planned Week 9-10]"

up:
	docker compose up -d

down:
	docker compose down

test:
	pytest

smoke:
	python scripts/run_twin_smoke_test.py

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf data/networks/sandbox
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

train:
	@echo "Not yet implemented: MARL training lands in Week 5-6 (see docs/design/13_roadmap.md)."
	@exit 1

eval:
	@echo "Not yet implemented: the evaluation harness lands in Week 11 (see docs/design/13_roadmap.md)."
	@exit 1

demo:
	@echo "Not yet implemented: the end-to-end demo lands in Week 9-10 (see docs/design/13_roadmap.md)."
	@exit 1
