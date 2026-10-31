"""Guards the delivery pipeline's configuration against the mistakes that are easy to make and
expensive to find late: a deploy job that isn't actually gated behind the tests, a production job
that could run without approval, a secret committed into an image or workflow, a Dockerfile that
runs as root, or an environment variable the code reads that nobody documented.

These are static checks over the files (no Docker, no GitHub needed), so they run in every CI
build and fail the PR that breaks the pipeline -- rather than the release that depends on it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from atlas.common.config import project_root

ROOT = project_root()
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} did not parse to a mapping"
    # YAML 1.1 reads the bare key `on` as the boolean True; normalise it.
    if True in data:
        data["on"] = data.pop(True)
    return data


CI = _load(WORKFLOWS / "ci.yml")
DEPLOY = _load(WORKFLOWS / "deploy.yml")
ALL_WORKFLOWS = sorted(WORKFLOWS.glob("*.yml"))


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


class TestWorkflowsParse:
    @pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
    def test_every_workflow_is_valid_yaml_with_triggers_and_jobs(self, path: Path) -> None:
        data = _load(path)

        assert data.get("on"), f"{path.name} has no triggers"
        assert data.get("jobs"), f"{path.name} has no jobs"

    @pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
    def test_every_workflow_declares_least_privilege_permissions(self, path: Path) -> None:
        # Without a top-level `permissions:` the GITHUB_TOKEN gets the repository default, which
        # is often read-WRITE on everything.
        assert "permissions" in _load(path), f"{path.name}: add a top-level permissions block"

    @pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
    def test_no_job_requests_write_access_it_does_not_need(self, path: Path) -> None:
        for name, job in _load(path)["jobs"].items():
            perms = job.get("permissions", {})
            if isinstance(perms, dict):
                writes = {k for k, v in perms.items() if v == "write"}
                assert writes <= {"packages"}, f"{path.name}:{name} requests write on {writes}"


class TestCiPipelineGraph:
    jobs = CI["jobs"]

    def test_every_needs_reference_points_at_a_real_job(self) -> None:
        for name, job in self.jobs.items():
            for dep in _needs(job):
                assert dep in self.jobs, f"{name} needs unknown job '{dep}'"

    def test_the_job_graph_has_no_cycles(self) -> None:
        resolved: set[str] = set()
        pending = dict(self.jobs)
        while pending:
            ready = [n for n, j in pending.items() if set(_needs(j)) <= resolved]
            assert ready, f"dependency cycle among {sorted(pending)}"
            for name in ready:
                resolved.add(name)
                del pending[name]

    def test_the_image_is_only_built_after_every_check_passes(self) -> None:
        # The core CI->CD guarantee: a failing test, lint, audit or frontend build blocks the image.
        assert set(_needs(self.jobs["docker"])) >= {
            "python-quality",
            "python-tests",
            "frontend",
            "security",
        }

    def test_ci_runs_on_every_push_and_pull_request(self) -> None:
        triggers = CI["on"]
        assert "pull_request" in triggers
        assert triggers["push"]["branches"] == ["**"]

    def test_tests_run_with_a_coverage_floor_and_build_the_network_first(self) -> None:
        steps = " ".join(str(s.get("run", "")) for s in self.jobs["python-tests"]["steps"])

        assert "--cov-fail-under" in steps
        assert "build_district_network.py" in steps  # else ~60 simulation tests silently skip

    def test_dependencies_are_installed_from_the_lockfile_only(self) -> None:
        for name in ("python-quality", "python-tests"):
            runs = [str(s.get("run", "")) for s in self.jobs[name]["steps"]]
            sync = next(r for r in runs if "uv sync" in r)
            assert "--frozen" in sync, f"{name} must not re-resolve dependencies"

    def test_the_image_is_smoke_tested_before_it_can_be_pushed(self) -> None:
        steps = self.jobs["docker"]["steps"]
        names = [s.get("name", "") for s in steps]

        smoke = next(i for i, n in enumerate(names) if n.startswith("Smoke-test the API"))
        push = next(i for i, n in enumerate(names) if n == "Push the tested images")
        assert smoke < push


class TestReleaseGating:
    jobs = CI["jobs"]

    @pytest.mark.parametrize("job", ["deploy-staging", "deploy-production"])
    def test_deploys_only_run_for_pushes_to_the_default_branch(self, job: str) -> None:
        condition = str(self.jobs[job]["if"])

        assert "github.event_name == 'push'" in condition
        assert "github.event.repository.default_branch" in condition

    def test_pull_requests_can_never_publish_an_image(self) -> None:
        steps = {s.get("name"): s for s in self.jobs["docker"]["steps"]}

        for name in ("Log in to GHCR", "Push the tested images"):
            assert "IS_RELEASE" in str(steps[name]["if"])
        release = str(self.jobs["docker"]["env"]["IS_RELEASE"])
        assert "github.event_name == 'push'" in release
        assert "default_branch" in release

    def test_staging_waits_for_the_image_and_production_waits_for_staging(self) -> None:
        assert "docker" in _needs(self.jobs["deploy-staging"])
        assert set(_needs(self.jobs["deploy-production"])) >= {"docker", "deploy-staging"}

    def test_production_has_an_explicit_off_switch(self) -> None:
        # A real production deploy must never be one merge away from happening by accident.
        assert "vars.PRODUCTION_DEPLOY_ENABLED == 'true'" in str(
            self.jobs["deploy-production"]["if"]
        )

    def test_both_environments_promote_the_same_image_tag(self) -> None:
        staging = self.jobs["deploy-staging"]["with"]["image_tag"]
        production = self.jobs["deploy-production"]["with"]["image_tag"]

        assert staging == production == "${{ needs.docker.outputs.image_tag }}"

    def test_deploys_reuse_the_one_deploy_workflow(self) -> None:
        for job in ("deploy-staging", "deploy-production"):
            assert self.jobs[job]["uses"] == "./.github/workflows/deploy.yml"
            assert self.jobs[job]["secrets"] == "inherit"

    def test_the_called_workflow_exists_and_receives_everything_it_requires(self) -> None:
        declared = DEPLOY["on"]["workflow_call"]["inputs"]
        required = {k for k, v in declared.items() if v.get("required")}
        for job in ("deploy-staging", "deploy-production"):
            given = set(self.jobs[job]["with"])
            assert required <= given, f"{job} is missing {required - given}"
            assert given <= set(declared), f"{job} passes undeclared inputs {given - set(declared)}"


class TestDeployWorkflow:
    job = DEPLOY["jobs"]["deploy"]
    steps = job["steps"]

    def test_it_targets_the_named_environment_so_protection_rules_apply(self) -> None:
        # This is what makes production's "required reviewers" pause the job for approval.
        assert self.job["environment"]["name"] == "${{ inputs.environment }}"

    def test_deploys_to_one_environment_never_overlap_or_cancel_midway(self) -> None:
        concurrency = DEPLOY["concurrency"]

        assert "inputs.environment" in concurrency["group"]
        assert concurrency["cancel-in-progress"] is False

    def test_manual_runs_can_deploy_an_older_tag_to_roll_back(self) -> None:
        inputs = DEPLOY["on"]["workflow_dispatch"]["inputs"]

        assert inputs["environment"]["options"] == ["staging", "production"]
        assert "roll back" in inputs["image_tag"]["description"].lower()

    def test_mutable_tags_are_refused(self) -> None:
        validate = next(s for s in self.steps if s.get("name") == "Validate inputs")

        assert "sha-[0-9a-f]" in validate["run"]

    def test_a_deployment_is_always_followed_by_a_smoke_test_of_the_new_version(self) -> None:
        names = [s.get("name", "") for s in self.steps]
        smoke = next(s for s in self.steps if s.get("name") == "Smoke test the deployment")

        assert names.index("Trigger deployment") < names.index("Smoke test the deployment")
        assert "--expect-version" in smoke["run"]
        assert "IMAGE_TAG" in smoke["run"]

    def test_an_unconfigured_environment_is_reported_not_faked(self) -> None:
        check = next(
            s for s in self.steps if s.get("name") == "Check this environment is configured"
        )
        guarded = [
            s["name"]
            for s in self.steps
            if "steps.cfg.outputs.configured == 'true'" in str(s.get("if", ""))
        ]

        assert "NOT DEPLOYED" in check["run"]
        assert {"Trigger deployment", "Smoke test the deployment"} <= set(guarded)

    def test_the_hook_url_secret_is_never_echoed(self) -> None:
        trigger = next(s for s in self.steps if s.get("name") == "Trigger deployment")
        run = trigger["run"]

        assert "> /dev/null" in run  # curl output (which could echo the URL) is discarded
        assert 'echo "${hook}' not in run
        assert "set -x" not in run


class TestNoSecretsInTheRepository:
    SECRET_PATTERNS = [
        r"ghp_[A-Za-z0-9]{20,}",  # GitHub personal access token
        r"gsk_[A-Za-z0-9]{20,}",  # Groq key
        r"AIza[0-9A-Za-z_-]{30,}",  # Google API key
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"https://[0-9a-f]{20,}@[a-z0-9.]*sentry\.io",  # Sentry DSN with key
    ]
    FILES = [
        *ALL_WORKFLOWS,
        ROOT / "Dockerfile",
        ROOT / "frontend" / "Dockerfile",
        ROOT / "docker-compose.yml",
        ROOT / ".env.example",
    ]

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_no_credential_shaped_strings(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")

        for pattern in self.SECRET_PATTERNS:
            assert not re.search(pattern, text), (
                f"{path.name} contains something matching {pattern}"
            )

    def test_env_example_has_no_values_for_secret_settings(self) -> None:
        secret_like = re.compile(r"(KEY|SECRET|TOKEN|DSN|PASSWORD)\b")
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            value = value.split("#", 1)[0].strip()
            if secret_like.search(name):
                assert value == "", f"{name} must be a blank placeholder in .env.example"

    def test_real_env_files_are_gitignored(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

        assert ".env" in ignored
        assert ".env.*" in ignored

    def test_the_only_jwt_secret_in_compose_is_the_documented_demo_default(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        values = set(re.findall(r"ATLAS_JWT_SECRET:\s*(.+)", compose))

        for value in values:
            assert value.startswith("${ATLAS_JWT_SECRET:-local-demo-secret-do-not-use"), value


class TestApiDockerfile:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]

    def _stage(self, name: str) -> list[str]:
        start = next(
            i for i, ln in enumerate(self.lines) if ln.upper().endswith(f"AS {name.upper()}")
        )
        rest = self.lines[start + 1 :]
        end = next((i for i, ln in enumerate(rest) if ln.upper().startswith("FROM ")), len(rest))
        return rest[:end]

    def test_it_is_multi_stage_and_the_runtime_has_no_build_tooling(self) -> None:
        runtime = "\n".join(self._stage("runtime"))

        assert len(re.findall(r"^FROM ", self.text, re.M)) == 2
        assert "uv " not in runtime.replace("--chown", "")
        assert "pip install" not in runtime

    def test_the_final_user_is_not_root(self) -> None:
        users = [ln for ln in self._stage("runtime") if ln.upper().startswith("USER ")]

        assert users, "runtime stage must drop privileges"
        assert users[-1].split()[1] not in ("root", "0")

    def test_no_secret_is_baked_into_the_image_environment(self) -> None:
        env_lines = [ln for ln in self.lines if ln.upper().startswith(("ENV ", "ARG "))]

        for line in env_lines:
            assert not re.search(r"(SECRET|PASSWORD|TOKEN|API_KEY|DSN)", line, re.I), line

    def test_it_declares_a_healthcheck_and_a_default_command(self) -> None:
        runtime = "\n".join(self._stage("runtime"))

        assert "HEALTHCHECK" in runtime
        assert re.search(r"^CMD \[", runtime, re.M)

    def test_dependencies_come_from_the_lockfile_and_exclude_dev_tooling(self) -> None:
        builder = "\n".join(self._stage("builder"))

        assert "uv sync --frozen --no-dev" in builder

    def test_the_road_network_is_built_inside_the_image(self) -> None:
        builder = "\n".join(self._stage("builder"))

        assert "build_district_network.py" in builder
        assert "generate_demand.py" in builder

    def test_the_build_context_excludes_secrets_and_stale_host_data(self) -> None:
        ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

        for entry in (".git/", ".env", ".venv/", "tests/", "frontend/"):
            assert entry in ignore
        # Only the committed OSM extract is taken from data/; a host-built network is not.
        assert "!data/networks/manhattan_midtown/raw.osm.xml" in ignore
        assert "!.env.example" in ignore


class TestFrontendDockerfile:
    text = (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")

    def test_it_serves_with_an_unprivileged_nginx_not_node(self) -> None:
        final = self.text.split("FROM ")[-1]

        assert final.startswith("nginxinc/nginx-unprivileged")
        assert "node" not in final.split("\n", 1)[0]

    def test_it_installs_exactly_the_lockfile(self) -> None:
        commands = [ln.strip() for ln in self.text.splitlines() if ln.strip().startswith("RUN ")]

        assert any(cmd.startswith("RUN npm ci") for cmd in commands)
        assert not any("npm install" in cmd for cmd in commands)  # (comments may mention it)

    def test_the_api_address_is_a_runtime_setting_not_baked_in(self) -> None:
        assert "VITE_API_BASE_URL" not in self.text
        assert "ATLAS_API_BASE_URL" in self.text
        template = (ROOT / "frontend" / "nginx.conf.template").read_text(encoding="utf-8")
        assert "${ATLAS_API_BASE_URL}" in template


class TestComposeFile:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    def test_the_app_profile_runs_api_dashboard_and_redis(self) -> None:
        in_app = {n for n, s in self.compose["services"].items() if "app" in s.get("profiles", [])}

        assert in_app == {"cognitive_api", "frontend", "redis"}

    @pytest.mark.parametrize(
        "service", ["cognitive_api", "frontend", "redis", "intersection_agent", "watchdog"]
    )
    def test_long_running_services_define_a_health_check(self, service: str) -> None:
        assert "healthcheck" in self.compose["services"][service]

    def test_the_dashboard_waits_for_a_healthy_api(self) -> None:
        depends = self.compose["services"]["frontend"]["depends_on"]

        assert depends["cognitive_api"]["condition"] == "service_healthy"

    def test_state_survives_restarts_through_a_named_volume(self) -> None:
        assert "atlas_state" in self.compose["volumes"]
        assert any(
            v.startswith("atlas_state:")
            for v in self.compose["services"]["cognitive_api"]["volumes"]
        )


class TestEnvironmentVariablesAreDocumented:
    # Variables the code reads that are deliberately internal and not meant for operators.
    INTERNAL = {"ATLAS_VERSION"}  # baked into the image by the build, not configured

    def _referenced(self) -> set[str]:
        names: set[str] = set()
        for base in (ROOT / "src", ROOT / "scripts"):
            for path in base.rglob("*.py"):
                names |= set(
                    re.findall(
                        r"[\"'](ATLAS_[A-Z0-9_]+|SENTRY_[A-Z0-9_]+)[\"']",
                        path.read_text(encoding="utf-8"),
                    )
                )
        return names

    def test_every_setting_the_code_reads_appears_in_env_example(self) -> None:
        example = (ROOT / ".env.example").read_text(encoding="utf-8")

        missing = sorted(n for n in self._referenced() - self.INTERNAL if n not in example)

        assert not missing, f"undocumented in .env.example: {missing}"
