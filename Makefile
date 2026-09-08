# ATLAS developer entrypoints. `make help` lists every target.

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
	@echo "  make train       - train MAPPO on the 5-junction corridor curriculum stage"
	@echo "  make eval        - reproduce the real Week 11 evaluation pilot report"
	@echo "  make demo        - run the Cognitive API demo server (for the frontend dashboard)"

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
	# Real, but not trained to convergence in this environment -- needs a real
	# Colab/Kaggle GPU session (see README's "What isn't done" / ADR-0013).
	python scripts/train_mappo_corridor.py

eval:
	python scripts/run_evaluation_pilot.py

demo:
	python scripts/run_dashboard_demo_api.py
