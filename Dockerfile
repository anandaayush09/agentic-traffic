# ATLAS control-plane services image (Week 6) -- shared by intersection_agent and
# watchdog, since both are plain `atlas` package entrypoints with the same dependencies.
# No system SUMO install needed: eclipse-sumo/libsumo/traci install as pip wheels on
# manylinux (verified for Windows in Week 1; the same wheels exist for linux/amd64).
#
# NOTE: Docker is not installed on this project's primary dev machine (see the ATLAS
# dev-environment memory) -- this build has not been run locally. Written correctly
# against uv's documented Docker recipe; verify with `docker build .` on a machine with
# Docker, or in CI, before relying on it.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY schemas/ schemas/
# net.net.xml/route files are gitignored build artifacts (scripts/build_district_network.py,
# scripts/generate_demand.py) -- must exist on the host BEFORE `docker build`, same
# prerequisite the local dev scripts already have.
COPY data/networks/ data/networks/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
