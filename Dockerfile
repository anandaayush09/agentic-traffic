# ATLAS API image (also the base for the control-plane services in docker-compose.yml:
# intersection_agent, watchdog, corridor_coordinator -- same package, different command).
#
# Two stages:
#   builder -- has uv and the build inputs; installs the locked dependencies and BUILDS the road
#              network + demand from the committed OSM extract. Nothing from here except the
#              results is shipped.
#   runtime -- python + the shared libraries SUMO needs + the venv, source and built network,
#              running as a non-root user. No uv, no pip cache, no build context leftovers.
#
# Why the network is built INSIDE the image: net.net.xml and the demand files are gitignored build
# artifacts. The previous image required them to exist on the host before `docker build`, so a
# clean checkout (or CI) could not build the image without a separate manual step, and the image
# silently depended on whatever happened to be on that machine. Building here makes the image a
# pure function of the repository: the same commit always produces the same image.
#
# Secrets are never baked in: configuration arrives as environment variables at run time
# (see .env.example), and the image defaults to ATLAS_ENV=development, which still refuses nothing
# -- staging/production deployments set ATLAS_ENV and the startup guard then refuses unsafe config.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

# libsumo's compiled extension links against a GUI-capable build's shared libs even for headless
# use (found by running the built image on a kind cluster: libX11, then libXext, were missing).
# netconvert, which builds the network below, needs the same set.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxext6 libxrender1 libsm6 libice6 libgl1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# uv is pinned so a new uv release can't change how the lockfile is interpreted under you.
RUN pip install --no-cache-dir uv==0.12.13

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, in their own layer: this layer is rebuilt only when pyproject.toml or
# uv.lock change, not on every source edit. `--frozen` refuses to re-resolve (the lockfile is the
# truth), `--no-dev` leaves test/lint tooling out, `--no-install-project` because the code is run
# from PYTHONPATH below, which also keeps README.md out of the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra observability --no-install-project

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
# Only the committed OSM extract; the root .dockerignore keeps any host-built network out of the
# build context so this stage can't accidentally "work" because of a stale local file.
COPY data/networks/manhattan_midtown/raw.osm.xml data/networks/manhattan_midtown/raw.osm.xml

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/src"
RUN python scripts/build_district_network.py \
    && python scripts/generate_demand.py \
    && test -s data/networks/manhattan_midtown/net.net.xml \
    && test -s data/networks/manhattan_midtown/demand/pm_peak.rou.xml

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxext6 libxrender1 libsm6 libice6 libgl1 libatomic1 \
    && rm -rf /var/lib/apt/lists/* \
    # Non-root (docs/design/10_deployment.md section 10.2). Fixed uid so file ownership is
    # predictable across rebuilds and volume mounts.
    && useradd --create-home --uid 10001 atlas \
    && mkdir -p /app/state && chown atlas:atlas /app /app/state

WORKDIR /app
# --chown on COPY, not a recursive chown afterwards: the latter would copy every file into a new
# layer and roughly double the image size.
COPY --from=builder --chown=atlas:atlas /app/.venv /app/.venv
COPY --from=builder --chown=atlas:atlas /app/src /app/src
COPY --from=builder --chown=atlas:atlas /app/configs /app/configs
COPY --from=builder --chown=atlas:atlas /app/scripts /app/scripts
COPY --from=builder --chown=atlas:atlas /app/data /app/data
COPY --chown=atlas:atlas schemas/ schemas/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ATLAS_LOG_FORMAT=json

# Which build is this? CI passes the commit-derived tag; `/version` reports it, and the
# post-deploy smoke test compares it to what was just shipped. Declared last so changing it
# doesn't invalidate any cached layer above.
ARG ATLAS_VERSION=dev
ENV ATLAS_VERSION=${ATLAS_VERSION}

USER atlas
EXPOSE 8000

# Liveness only (the process answers). /ready is the readiness gate used by deploys. Services
# that listen elsewhere (watchdog, coordinator) override this in docker-compose.yml.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"]

CMD ["python", "scripts/run_dashboard_demo_api.py"]
