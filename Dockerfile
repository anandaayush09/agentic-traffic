# ATLAS control-plane services image (Week 6) -- shared by intersection_agent,
# watchdog, corridor_coordinator, and (Week 10) the Cognitive API demo server, since
# all are plain `atlas` package entrypoints with the same dependencies. No system SUMO
# install needed: eclipse-sumo/libsumo/traci install as pip wheels on manylinux
# (verified for Windows in Week 1; the same wheels exist for linux/amd64).
#
# Verified with a real `docker build .` (Week 10, once Docker became available on this
# dev machine -- see the ATLAS dev-environment memory's 2026-09-07 update) --
# previously written correctly against uv's documented recipe but unverified locally.
FROM python:3.12-slim

# libsumo's compiled extension links against a GUI-capable build's shared libs even
# for headless use -- caught iteratively for real running the built image on a kind
# cluster (libX11 missing, then libXext) rather than guessed up front. Installing the
# fuller standard X11/GL runtime set together (the same list opencv/matplotlib-on-
# headless-Linux images typically need) to avoid a third build-deploy-diagnose cycle
# for the next one of these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxext6 libxrender1 libsm6 libice6 libgl1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
# README.md must be copied before `uv sync`, not after: pyproject.toml declares
# readme = "README.md", and hatchling's editable-install build validates that file
# exists at build time -- caught by a real `docker build .` failure ("OSError: Readme
# file does not exist: README.md"), not assumed to work from inspection alone.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY schemas/ schemas/
COPY scripts/ scripts/
# net.net.xml/route files are gitignored build artifacts (scripts/build_district_network.py,
# scripts/generate_demand.py) -- must exist on the host BEFORE `docker build`, same
# prerequisite the local dev scripts already have.
COPY data/networks/ data/networks/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Non-root, per docs/design/10_deployment.md section 10.2 ("non-root user") --
# everything above runs as root only because apt-get/uv need it; the actual service
# process never does. chown after every COPY so the app user can still read its own
# files (the .venv/src/schemas/scripts/data trees are all owned by root by default).
RUN useradd --create-home --uid 10001 atlas && chown -R atlas:atlas /app
USER atlas
