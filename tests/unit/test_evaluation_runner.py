"""Validates run_scenario_cell against the real sandbox network and real SUMO -- no
mocks. Confirms it produces sane, real TrafficMetrics for every non-RL controller
mode, that a scenario applied mid-episode is genuinely exercised (not silently
ignored), and that mode="rl" fails loudly rather than silently, matching the
runner's own documented scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.common.config import project_root
from atlas.evaluation.runner import run_scenario_cell
from atlas.twin.scenario import ScenarioSpec

NET_FILE = project_root() / "data" / "networks" / "sandbox" / "net.net.xml"
ROUTE_FILE = project_root() / "data" / "networks" / "sandbox" / "routes.rou.xml"


class TestRunScenarioCell:
    @pytest.mark.parametrize("mode", ["max_pressure", "fixed_time", "actuated"])
    def test_produces_real_sane_metrics_for_every_non_rl_mode(
        self, mode: str, tmp_path: Path
    ) -> None:
        metrics = run_scenario_cell(
            mode=mode,  # type: ignore[arg-type]
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
            tripinfo_output=tmp_path / f"tripinfo_{mode}.xml",
        )

        assert metrics.num_trips > 0
        assert metrics.mean_waiting_time_s >= 0
        assert metrics.mean_travel_time_s > 0
        assert metrics.throughput_veh_per_hour >= 0
        assert metrics.mean_queue_length >= 0
        assert metrics.p95_queue_length >= metrics.mean_queue_length

    def test_same_seed_gives_reproducible_metrics(self, tmp_path: Path) -> None:
        kwargs = dict(
            mode="max_pressure",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=7,
            num_decisions=15,
        )
        first = run_scenario_cell(tripinfo_output=tmp_path / "a.xml", **kwargs)
        second = run_scenario_cell(tripinfo_output=tmp_path / "b.xml", **kwargs)

        assert first == second

    def test_a_scenario_spec_is_actually_applied_during_the_episode(self, tmp_path: Path) -> None:
        """A closure on B1's own controlled edge for the WHOLE episode should measurably
        change outcomes vs. no scenario at all -- proving scenario_specs is genuinely
        wired into the runner's loop, not accepted and ignored."""
        without_scenario = run_scenario_cell(
            mode="max_pressure",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
            tripinfo_output=tmp_path / "no_scenario.xml",
        )
        with_closure = run_scenario_cell(
            mode="max_pressure",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
            tripinfo_output=tmp_path / "with_closure.xml",
            scenario_specs=[
                ScenarioSpec(
                    name="whole_episode_closure",
                    type="closure",
                    edge_id="A1B1",
                    start_s=0,
                    duration_s=1_000_000,
                )
            ],
        )

        assert with_closure != without_scenario

    def test_detector_dropout_is_actually_wired_into_the_decision_loop(
        self, tmp_path: Path
    ) -> None:
        """Full (100%) detector dropout for a queue-sensitive controller
        (max_pressure) must change its decisions vs. no dropout -- proving
        detector_dropout_rate is genuinely applied, not accepted and ignored."""
        kwargs = dict(
            mode="max_pressure",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
        )
        without_dropout = run_scenario_cell(tripinfo_output=tmp_path / "no_dropout.xml", **kwargs)
        with_full_dropout = run_scenario_cell(
            tripinfo_output=tmp_path / "full_dropout.xml",
            detector_dropout_rate=1.0,
            **kwargs,
        )

        assert with_full_dropout != without_dropout

    def test_zero_dropout_rate_is_a_true_no_op(self, tmp_path: Path) -> None:
        kwargs = dict(
            mode="max_pressure",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
        )
        explicit_zero = run_scenario_cell(
            tripinfo_output=tmp_path / "explicit_zero.xml",
            detector_dropout_rate=0.0,
            **kwargs,
        )
        default = run_scenario_cell(tripinfo_output=tmp_path / "default.xml", **kwargs)

        assert explicit_zero == default

    def test_fixed_time_is_unaffected_by_detector_dropout(self, tmp_path: Path) -> None:
        """fixed_time never reads queue_lengths at all -- this is the honest,
        expected result of the robustness sweep for that mode, not a bug."""
        kwargs = dict(
            mode="fixed_time",
            tls_ids=["B1"],
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            seed=42,
            num_decisions=20,
        )
        without_dropout = run_scenario_cell(
            tripinfo_output=tmp_path / "ft_no_dropout.xml", **kwargs
        )
        with_full_dropout = run_scenario_cell(
            tripinfo_output=tmp_path / "ft_full_dropout.xml",
            detector_dropout_rate=1.0,
            **kwargs,
        )

        assert with_full_dropout == without_dropout

    def test_rl_mode_raises_rather_than_silently_producing_garbage(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="net and obs"):
            run_scenario_cell(
                mode="rl",
                tls_ids=["B1"],
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                seed=42,
                num_decisions=5,
                tripinfo_output=tmp_path / "rl.xml",
            )
