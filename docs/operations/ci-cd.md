# CI/CD, Docker and operations guide

How code gets from your laptop to a running, verified deployment, and how to run, watch and roll back
that deployment. Written to teach the concepts as well as document the commands.

- [The big picture](#the-big-picture)
- [Concepts in 2 minutes](#concepts-in-2-minutes)
- [What each pipeline job does](#what-each-pipeline-job-does)
- [One-time setup (what only you can do)](#one-time-setup-what-only-you-can-do)
- [Environment variables and secrets](#environment-variables-and-secrets)
- [Releasing: staging, then production](#releasing-staging-then-production)
- [Verify the whole process end to end](#verify-the-whole-process-end-to-end)
- [Rollback](#rollback)
- [Logs and monitoring](#logs-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [What was and wasn't verified](#what-was-and-wasnt-verified)

## The big picture

```
 you: git push
        |
        v
 +----------------------------- CI (every push + every pull request) -----------------------------+
 |  python-quality   python-tests        frontend          security                                |
 |  ruff, format,    pytest + coverage   eslint, tsc,      pip-audit (locked deps),                |
 |  mypy --strict    (real SUMO sim)     vite build,        npm audit                              |
 |                                       npm audit                                                 |
 |        \               |                  |                /                                   |
 |         +--------------+---- all must pass ------+--------+                                    |
 |                                |                                                                |
 |                                v                                                                |
 |   docker: build API + web images -> run the container -> smoke test it -> (release only) push  |
 +-------------------------------------------|----------------------------------------------------+
                                             | only for pushes to the DEFAULT branch
                                             v
 +----------------------------- CD (deploy.yml, reusable) ---------------------------------------+
 |  staging:  verify image in registry -> trigger deploy -> smoke test the NEW version             |
 |                                             | passes                                            |
 |                                             v                                                   |
 |  production:  [ manual approval ]  -> same image, same steps -> smoke test                      |
 +------------------------------------------------------------------------------------------------+
```

**CI** (Continuous Integration) answers "is this change safe to merge?" It runs on every push and PR and
never touches a real environment. **CD** (Continuous Delivery) answers "get the merged, tested thing to
users". It runs only for the default branch, only after CI passed, and promotes *the same image* CI built:
nothing is rebuilt for staging or production, so what runs in production is exactly what was tested.

If any CI job fails, the `docker` job never starts, so there is no image, so the deploy jobs never start.
That chain of `needs:` in `.github/workflows/ci.yml` is the whole mechanism that makes "a failing test
blocks deployment" true.

## Concepts in 2 minutes

| Term | What it is here |
|---|---|
| **Workflow** | A YAML file in `.github/workflows/`. `ci.yml` (the pipeline), `deploy.yml` (deploy + rollback), `kind.yml` (Kubernetes smoke test on PRs), `eval.yml` (nightly regression). |
| **Job / step** | A job is a group of steps that runs on one machine; jobs run in parallel unless one `needs:` another. A step is a shell command or a reusable *action*. |
| **Runner** | The machine that executes a job. GitHub provides them (`ubuntu-latest`): a fresh, empty VM per job, which is why every job starts with `checkout` + install. GitHub *manages* the pipeline (decides what runs, when, in what order, stores secrets and logs); the *runner* just executes the commands it is handed. |
| **Docker image** | A frozen, read-only package: OS libraries + Python + the app + the built road network. Built once from the `Dockerfile`. |
| **Container** | A running instance of an image. Start as many as you like from one image; they are disposable. |
| **Registry** | A server that stores images. Here GitHub's **GHCR** (`ghcr.io`), free for this repo. CI pushes; the hosting platform pulls. |
| **Image tag** | A label on an image. We use `sha-<12 chars of the commit>`: immutable and traceable (tag -> commit -> pipeline run). We never deploy `latest` because it moves and "what is running?" would become unanswerable. |
| **Environment** | A GitHub setting (`staging`, `production`) that holds that stage's secrets/variables and protection rules. A job that targets `production` is the *only* one that can read production secrets. |
| **Smoke test** | A fast check after deploying that the thing actually works (see below). Distinct from the test suite: tests check the code, the smoke test checks the *deployment*. |

### Where Docker is used in this project

1. **`Dockerfile`** (API) is multi-stage. The *builder* stage installs the locked dependencies and builds the
   road network from the committed OSM file; the *runtime* stage copies only the results into a slim image that
   runs as a non-root user. Build tools (uv, pip caches) never reach the final image.
2. **`frontend/Dockerfile`**: node builds the React app, then **nginx** (unprivileged) serves the static files.
   The API address is a *runtime* setting (`ATLAS_API_BASE_URL`, served as `/config.js`), so one image works in
   every environment.
3. **`docker-compose.yml`**: runs several containers together locally (API + dashboard + Redis).
4. **CI** builds both images on every push/PR, starts the API container and runs the smoke test *against the
   container*, so a broken Dockerfile or a container that won't start fails the PR, not the release.

## What each pipeline job does

| Job | Part | What it checks | Fails when |
|---|---|---|---|
| `python-quality` | CI | `ruff check` (lint **and** bandit-style security rules), `ruff format --check`, `mypy --strict` | style/type/security-lint errors |
| `python-tests` | CI | 800+ tests (unit + integration) with a **90% coverage floor**. Builds the road network first so ~60 real-simulation tests actually *run* instead of silently skipping. LLM providers and Redis are mocked: no keys, no paid calls | a test fails, coverage drops below 90%, or the network is missing |
| `frontend` | CI | `npm ci`, eslint, `tsc` + `vite build`, `npm audit` (high+) | lint/type/build error or a high-severity npm advisory |
| `security` | CI | `pip-audit` over the *exact locked* versions | a known vulnerability in a locked dependency |
| `docker` | CI (+ publish on release) | builds both images, checks the API container is non-root, **runs it and smoke-tests it**, scans it with Trivy (report-only), and **only on the default branch** pushes both images tagged `sha-...` | build fails, container won't start, smoke test fails |
| `deploy-staging` | **CD** | verifies the image exists, triggers the deploy, waits for the **new** version to answer, runs the smoke test | deploy trigger fails or the smoke test fails |
| `deploy-production` | **CD** | same, after staging passed **and** a human approved | as above |

The **smoke test** (`scripts/smoke_test_deployment.py`) checks: `/health` is up, `/ready` (simulation warmed
up), `/version` equals the commit just shipped (a rolling deploy keeps the *old* build answering for a while, so
a bare "200 OK" can be the old version passing), the incident list works, live traffic has vehicles, an
unauthenticated `POST /incidents` is **rejected (401)**, and the no-login token endpoint is **hidden (404)**.
The last two exist because a deployment that accidentally turned auth off is "healthy" by every other measure.

## One-time setup (what only you can do)

Nothing real is deployed until you do this. Until then the staging job reports
**"NOT DEPLOYED: environment is not configured"** in its summary: it never pretends.

### 0. Make the pipeline run at all
This repository has **no `main` branch**; its default is `week11/evaluation-suite`. The old workflows only
triggered on `main`, so they never ran. The new ones run on every branch and treat *the repository's actual
default branch* as the release branch. When you later create `main` and make it the default
(Settings -> Branches), nothing in the workflows needs to change.

### 1. Pick a host. Recommendation: **Render**
ATLAS is a long-running, stateful process (a background traffic simulation, a WebSocket stream, in-memory/SQLite
state), which rules out platforms that freeze a container between requests.

| Option | Verdict |
|---|---|
| **Render** (Docker web service from a registry image) | **Recommended**: always-on container, WebSockets work, deploy hook triggers a pull of a specific image tag, simple. Use a paid instance (the free tier sleeps, which kills the simulation). Size the instance by measuring (`docker stats` locally): the API loads PyTorch, so assume >= 1 GB RAM. |
| Fly.io | Good alternative (`flyctl deploy --image`); similar cost. |
| Google Cloud Run | Workable but needs *min-instances=1* and *CPU always allocated* (otherwise CPU is throttled between requests and the simulation stalls), which removes its cost advantage; WebSocket sessions are capped at 60 min. |
| AWS (ECS/Fargate) | Fully capable, but the most setup (VPC, IAM, load balancer) for a project this size. |
| Kubernetes | A Helm chart exists in `deploy/helm/atlas` (used by the kind smoke test). Worth it only if you already run a cluster. |

Only the **Trigger deployment** step in `deploy.yml` is platform-specific; switching hosts means replacing that
one step (e.g. `gcloud run deploy --image ...`, `flyctl deploy --image ...`).

### 2. Create the services (staging first)
1. Make the GHCR packages readable by Render: after the first release publishes `atlas-api` and `atlas-web`,
   either set them to *public* (Package settings) or give Render a registry credential (a GitHub token with
   `read:packages`) when you create the service.
2. In Render: **New -> Web Service -> Deploy an existing image**, image `ghcr.io/<owner>/atlas-api:<any
   published sha tag>`, health check path `/ready`. Repeat for the dashboard with `atlas-web` (port 8080,
   health check `/healthz`). Do staging and production as separate services.
3. Set each service's environment variables (see the next section).
4. In each Render service: Settings -> **Deploy Hook** -> copy the URL (it contains a secret key).

### 3. Configure GitHub
Settings -> **Environments** -> create `staging` and `production`. In each:

| Kind | Name | Value |
|---|---|---|
| Secret | `DEPLOY_HOOK_API` | the API service's deploy hook URL |
| Secret | `DEPLOY_HOOK_WEB` | *(optional)* the dashboard's deploy hook URL |
| Variable | `APP_URL` | the API's public URL, e.g. `https://atlas-api-staging.onrender.com` |
| Variable | `WEB_URL` | *(optional)* the dashboard's public URL |

For **`production`** also: *Required reviewers* -> add yourself. GitHub then pauses the production job until you
click **Approve** (this is the manual approval gate). Then, only when you actually want production deploys:
Settings -> Secrets and variables -> Actions -> **Variables** -> add repository variable
`PRODUCTION_DEPLOY_ENABLED` = `true`. Until it exists the production job doesn't even start, a deliberate extra
lock on top of the reviewer gate.

> Environment secrets are only readable by jobs that name that environment, so the staging job can never read
> production credentials. The deploy hook is the *only* credential the pipeline holds, and it can do one thing:
> deploy an image to one service (least privilege). CI itself uses the automatic `GITHUB_TOKEN`, scoped to
> exactly `contents: read` (+ `packages: write` in the `docker` job only).

### 4. Optional but recommended
- **Branch protection** (Settings -> Branches -> add rule for the default branch): require pull requests and
  require the status checks `Python lint, format, types`, `Python tests (unit + integration)`,
  `Frontend lint + build`, `Dependency vulnerability audit` and `Build, test and publish images` to pass. This
  is what makes CI *block merging* rather than just report.
- **Sentry**: create a free project, copy its DSN into the services' `SENTRY_DSN`.

## Environment variables and secrets

Everything is configured through environment variables; `.env.example` documents each and contains
**placeholders only** (`.env` is gitignored; a test fails the build if a secret-shaped value is committed).

| Variable | Where it must be set | Notes |
|---|---|---|
| `ATLAS_ENV` | API service | `staging` / `production`. **Turns on the startup guard** (below). |
| `ATLAS_JWT_SECRET` | API service (**secret**) | Required outside development. >= 32 chars and not the public demo value. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Use a *different* one per environment. |
| `ATLAS_DASHBOARD_ORIGINS` | API service | The dashboard's exact URL, e.g. `https://atlas-staging-web.onrender.com`. Required outside development (CORS + WebSocket origin check). |
| `ATLAS_API_BASE_URL` | dashboard service | The API's public URL; the browser calls it. |
| `ATLAS_LOG_FORMAT` | any | `json` (the image default) for searchable logs. |
| `PORT` | set by the platform | The API listens on `$PORT` (default 8000). |
| `SENTRY_DSN` | API service (secret) | Optional. Sentry is off unless set. |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | API service (**secrets**) | Optional. Without them proposals come from the deterministic mapper. Never put them in the repo, Docker image or chat. |
| `ATLAS_REDIS_URL` | API service | Optional shared rate-limit state; falls back to in-memory. |
| `ATLAS_STATE_DB` | API service | Optional; keeps incident history across restarts (needs a persistent disk). |
| `DEPLOY_HOOK_API/WEB`, `APP_URL`, `WEB_URL` | GitHub **Environment** | See [setup](#3-configure-github). |

**The startup guard.** In `staging`/`production` the API refuses to boot (exit code 2, with a list of reasons) if
`ATLAS_JWT_SECRET` is missing, is the public demo value from this repository, or is under 32 characters; if
`ATLAS_DEMO_AUTH=1` (a no-login token endpoint); or if `ATLAS_DASHBOARD_ORIGINS` is unset. A service that
would otherwise come up "healthy" while signing tokens with a publicly-known secret fails loudly instead. It
also **never prints an operator token to the logs** outside development. Mint one deliberately:

```bash
ATLAS_JWT_SECRET=<that environment's secret> python scripts/mint_operator_token.py --subject you --hours 1
```

## Releasing: staging, then production

1. Open a pull request. CI runs; fix anything red.
2. Merge to the default branch (a normal push to it triggers the release half).
3. CI builds the images, tests the container, and publishes `ghcr.io/<owner>/atlas-api:sha-<commit>` (and `-web`).
4. `deploy-staging` runs automatically: it checks the tag exists, triggers the deploy, **waits for `/version` to
   report that exact tag**, and smoke-tests. Watch it in the Actions tab; the run summary shows the result.
5. Check staging by hand (open the dashboard; submit a demo accident; approve it).
6. `deploy-production` waits for your **Approve** click (if you've done the setup above). Same image, same steps.

You never build a "production image". The thing you approve is the thing you tested.

## Verify the whole process end to end

A checklist to prove each link works, in increasing order of commitment:

1. **Locally**: `docker compose --profile app up --build`; open http://localhost:8080. Then
   `python scripts/smoke_test_deployment.py --base-url http://localhost:8003 --require-live-traffic`
   (the compose API is published on 8003) should print `ALL CHECKS PASSED`.
2. **CI**: push a branch, open a PR, watch all jobs go green in the PR's *Checks* tab.
3. **A failing check blocks things**: on a throwaway branch, break a test on purpose, push, and confirm the
   `python-tests` job fails, `docker` shows *skipped*, and no image is published.
4. **Publishing**: merge to the default branch; confirm the image appears under the repo's *Packages* with
   the `sha-...` tag.
5. **Staging**: with the staging environment configured, confirm the `deploy-staging` summary says
   "Deployed", and `curl https://<staging api>/version` returns that tag.
6. **Production**: confirm the job pauses for approval; approve; confirm `/version` matches.
7. **Rollback drill** (do this *before* you need it): see below.

## Rollback

A deploy is just "run image tag X", so rolling back is "deploy the previous tag", through the *same* verified
path, not a separate script that might have rotted.

1. Find the last good tag: the Actions run summary of the previous successful deploy (it prints the tag), or
   the repo's *Packages* page, or `git log` (tag = `sha-` + the first 12 chars of the commit).
2. Actions tab -> **Deploy** -> **Run workflow** -> pick the environment, set `image_tag` to that
   `sha-...` value -> Run. (Production still pauses for approval; approve it.)
3. The workflow verifies the image exists, redeploys, and smoke-tests that the **old** version is the one
   answering. A green run means you are verifiably back on the old build.
4. Then fix forward on a branch; don't try to "revert" by editing the deployed environment.

If the platform itself is wedged, use its own dashboard's *Rollback/Redeploy previous* as a fallback; the
registry keeps every `sha-` image, so any previous build is always available.

Things a rollback does **not** undo: incident history written during the bad release (kept in the
`ATLAS_STATE_DB` file, if enabled) and any change you made in the platform's dashboard.

## Logs and monitoring

- **CI job logs**: repo -> *Actions* -> the run -> click a job -> click a step. Failed steps are expanded and
  red. Download everything: the run page's gear icon -> *Download log archive*. Test and coverage reports are
  attached to the run as the `python-test-reports` artifact. A failing `docker` job prints the container's
  own logs in its *Container logs* step.
- **Container logs**: `docker logs -f atlas-cognitive-api` (compose: `docker compose logs -f cognitive_api`).
  On Render: the service's *Logs* tab. In the image, logs are **one JSON object per line**
  (`ATLAS_LOG_FORMAT=json`), so you can filter them, e.g.
  `docker logs atlas-cognitive-api 2>&1 | grep '^{' | jq 'select(.level=="error")'`.
  Every unhandled error is logged as `api.unhandled_error` with its stack trace and request path; the client
  only ever sees a generic `500`.
- **Health endpoints**: `/health` (alive; used by container health checks), `/ready` (warmed up and fit for
  traffic; used by deploy gates; returns 503 until then), `/version` (which build is running).
- **Sentry** (optional, free tier): set `SENTRY_DSN`. Errors then arrive tagged with `release=sha-...` and the
  environment, so you can see which deploy introduced a problem. No PII and no request bodies are sent.
  Nothing else (no paid APM) is required.
- **Image vulnerabilities**: the Trivy table in the `docker` job's log; Dependabot opens PRs weekly for
  Python/npm/Actions/base-image updates (each runs through this same CI).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| CI never runs on my push | The workflow must exist on the branch you pushed; check *Actions* isn't disabled for the repo. (The old config only triggered on a non-existent `main`.) |
| `python-tests` fails with *"net.net.xml"* | The road-network build step failed or its cache is stale: open the *Build road network* step. Re-run with the cache cleared (Actions -> Caches -> delete `network-...`). |
| `python-tests`: *coverage below 90%* | You added code without tests, or tests skipped. Read the `--cov-report=term-missing` table in the log. |
| `security` fails on a new advisory | `uv lock --upgrade-package <pkg>`, re-run tests, push. If there is no fix yet and you've assessed it as not exploitable: add `--ignore-vuln <ID>` to the step with a comment explaining why. |
| `docker` fails at *Build API image* | Open the failing `RUN` line. Common: a new Python dependency needs a system library (add it to **both** `apt-get` lists in the `Dockerfile`), or `uv.lock` is out of sync (`uv lock`). |
| `docker` fails at *Smoke-test the API container* | The *Container logs* step shows why. A startup-guard refusal prints `Refusing to start in ATLAS_ENV=...` plus reasons. |
| Container starts then exits (exit 2) | The startup guard (see above): set the listed variables. |
| Dashboard loads but can't reach the API | `ATLAS_API_BASE_URL` (dashboard) or `ATLAS_DASHBOARD_ORIGINS` (API) is wrong. They must be each other's exact origin, scheme included, no trailing slash. |
| `deploy-*` says **NOT DEPLOYED** | That environment lacks `DEPLOY_HOOK_API` or `APP_URL`; expected until you finish setup. |
| Smoke test: *running 'sha-OLD', expected 'sha-NEW'* | The platform hasn't switched yet (the test keeps waiting up to 15 min) or never accepted the image. Check the service's deploy log; confirm it can pull from GHCR (package visibility/credential). |
| Smoke test: *auth enforced ... FAIL* | **Treat as a security incident**: the API accepted an unauthenticated write. Roll back first, investigate second. |
| `docker compose up` can't write `/app/state` | An old volume with wrong ownership: `docker compose down -v` (this deletes saved incident history). |
| `uv sync --frozen` fails in CI | `pyproject.toml` and `uv.lock` disagree. Run `uv lock` locally and commit the lockfile. |
| Windows: DLL error importing libsumo | `sqlite3` was imported before libsumo (SUMO ships its own `sqlite3.dll`). Already handled in `tests/native_preload.py`; for your own scripts `import libsumo` first. |

## What was and wasn't verified

Verified (run for real on the development machine):
- Lint, format, strict type check, the full test suite (824 tests, 94% coverage), frontend lint and build.
- The runtime dependency set: replayed the Dockerfile's builder steps natively in a clean directory (locked
  runtime dependencies only; road network and demand built from just the committed OSM file in ~35 s).
- The API started from that clean environment in `staging` mode: it **refuses** an unsafe configuration;
  with a safe one it passed all 7 smoke checks, printed no operator token, emitted JSON logs, accepted a
  freshly minted token and rejected a forged one; the smoke test correctly **failed** on a wrong version.
- `docker compose config` validates for the `app` and `full` profiles; every workflow parses and passes the
  structural tests in `tests/unit/test_deployment_config.py` (job graph, release gating, no secrets, Dockerfile
  safety, documented environment variables).
- The dependency audit: 0 known vulnerabilities (it found PyJWT, urllib3, langgraph and others, now upgraded).

**Not verified (could not be, here):**
- `docker build` / `docker run`: the Docker daemon was not running on the development machine. The first CI run
  is the first real build; expect to possibly fix a Dockerfile detail there.
- The workflows have never run on GitHub (no push was made), and `actionlint` could not be installed.
- No deployment of any kind: no host, registry credentials or deploy hooks exist yet. The Render deploy-hook
  call (`imgURL=` parameter) follows Render's documented format as I know it, so confirm it when you create
  the service.
- Trivy, `npm audit` in CI, branch protection and environment approval rules (GitHub-side settings).
- Memory sizing of the production container (measure with `docker stats`).
