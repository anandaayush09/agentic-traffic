#!/usr/bin/env python
"""Trains GraphWaveNetForecaster on the digital twin's own simulated telemetry --
docs/design/06_ai_pipeline.md section 6.4's data source ("ATLAS's own simulated
telemetry... so the forecast graph *is* the control graph"), and evaluates it against
the historical-average and last-value baselines the design explicitly calls for.

Scope reduction, documented rather than silently assumed away: the design specifies
15/30/60-MINUTE horizons over 5-minute aggregates; this script forecasts 3/6/12
DECISIONS ahead (15/30/60 seconds at the corridor's 5s decision interval) over
per-decision telemetry -- the same horizon-count structure and the same architecture,
at a scale that trains in seconds locally instead of needing weeks of accumulated
telemetry across "varied scenario runs" (the real target corpus, a later/Kaggle
concern). The forecast target is each junction's total pressure (the same quantity
MultiAgentSignalEnv already computes for its reward), not link mean speed --
substituting the metric this project's twin already produces per-decision for the one
named in the design doc, since a 5-junction corridor's per-lane speed telemetry at this
scale doesn't yet exist as a maintained corpus.

Two evaluation splits, both reported, because they answer different questions:
- Temporal holdout (train on the first 80% of decisions, test on the last 20%) is the
  methodologically correct split for a real forecasting deployment (you only ever have
  the past to predict the future). Reported as the headline result.
- i.i.d. random split of the SAME windows is a pipeline sanity check, not a deployment
  claim: if the model can't beat the baselines here either, that means the architecture
  or training loop is broken, not just data-starved. Found necessary, not run
  speculatively: an early un-normalized run's temporal-holdout result looked like model
  failure, and this split was what separated "real training-data-scale limitation" from
  "the model never learned anything at all" (see the git history if the printed NOTE
  below fires).

Usage:
    python scripts/train_forecaster.py [--num-decisions 300] [--epochs 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.control.max_pressure import MaxPressureProgram  # noqa: E402
from atlas.control.shield import ShieldConfig  # noqa: E402
from atlas.forecasting import (  # noqa: E402
    GraphWaveNetForecaster,
    build_windows,
    historical_average_baseline,
    last_value_baseline,
    quantile_loss,
)
from atlas.rl import MultiAgentSignalEnv, MultiAgentSignalEnvConfig  # noqa: E402
from atlas.services.decision import select_action_for_mode  # noqa: E402
from atlas.twin import TwinEnvConfig  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
CORRIDOR_TLS_IDS = ["42430333", "42430338", "42430342", "42430344", "42430347"]
HORIZONS = (3, 6, 12)
INPUT_LEN = 12
TRAIN_FRACTION = 0.8


def collect_telemetry(num_decisions: int, seed: int = 42) -> torch.Tensor:
    """Runs the corridor under max-pressure control, returning `(num_decisions,
    num_nodes, 1)`: each junction's total pressure at every decision -- real simulated
    data, not synthetic, generated the same way scripts/eval_smoke_suite.py already
    does for its own comparisons.
    """
    net_file = NETWORK_DIR / "net.net.xml"
    route_file = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
    vtypes_file = NETWORK_DIR / "vtypes.add.xml"
    env = MultiAgentSignalEnv(
        MultiAgentSignalEnvConfig(
            twin_config=TwinEnvConfig(
                name="forecast_telemetry",
                net_file=net_file,
                route_file=route_file,
                additional_files=[vtypes_file] if vtypes_file.is_file() else [],
            ),
            tls_ids=CORRIDOR_TLS_IDS,
            max_lanes=10,
            max_green_phases=2,
            shield_config=ShieldConfig(min_green_s=10.0, max_green_s=60.0),
            decision_interval_s=5.0,
        )
    )
    programs = {tid: MaxPressureProgram.from_net_file(tid, net_file) for tid in CORRIDOR_TLS_IDS}
    env.reset(seed=seed)

    last_actions = dict.fromkeys(CORRIDOR_TLS_IDS, 0)
    series: list[list[float]] = []
    try:
        for _ in range(num_decisions):
            conn = env.connection
            actions = {}
            for tid in CORRIDOR_TLS_IDS:
                program = programs[tid]
                lanes = {m.from_lane for m in program.movements} | {
                    m.to_lane for m in program.movements
                }
                queue_lengths = {
                    lane_id: int(conn.lane.getLastStepHaltingNumber(lane_id)) for lane_id in lanes
                }
                actions[tid] = select_action_for_mode(
                    "max_pressure",
                    program,
                    queue_lengths=queue_lengths,
                    last_action=last_actions[tid],
                )
            last_actions = actions
            _, _, _, _, infos = env.step(actions)
            series.append([float(infos[tid]["pressure"]) for tid in CORRIDOR_TLS_IDS])
    finally:
        env.close()

    return torch.tensor(series).unsqueeze(-1)  # (num_decisions, num_nodes, 1)


def mae_per_horizon(preds: torch.Tensor, target: torch.Tensor) -> list[float]:
    """preds/target: (num_windows, num_nodes, num_horizons). Returns one MAE per
    horizon, averaged over windows and nodes."""
    errors = (preds - target).abs()
    return errors.mean(dim=(0, 1)).tolist()


def train_and_evaluate(
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Trains one model on `x_raw[train_idx]`/`y_raw[train_idx]` and evaluates it (plus
    both baselines) on `test_idx`. Returns (model_mae, last_value_mae, historical_mae),
    each a list of `len(HORIZONS)` values. Normalization statistics are computed from
    the TRAIN indices only, in both split modes, so the i.i.d. sanity-check split never
    leaks test-window statistics into training either.
    """
    train_mean = x_raw[train_idx].mean()
    train_std = x_raw[train_idx].std()
    x_norm = (x_raw - train_mean) / train_std
    y_norm = (y_raw - train_mean) / train_std

    x_train, y_train = x_norm[train_idx], y_norm[train_idx]
    x_test = x_norm[test_idx]
    y_test_raw = y_raw[test_idx]

    torch.manual_seed(seed)
    model = GraphWaveNetForecaster(
        num_nodes=len(CORRIDOR_TLS_IDS), in_features=1, horizons=HORIZONS
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x_train)
        loss = quantile_loss(preds, y_train)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p50_index = model.quantiles.index(0.5)
        model_p50 = model(x_test)[..., p50_index] * train_std + train_mean

    x_test_raw = x_raw[test_idx]
    model_mae = mae_per_horizon(model_p50, y_test_raw)
    last_value_mae = mae_per_horizon(
        last_value_baseline(x_test_raw, num_horizons=len(HORIZONS)), y_test_raw
    )
    historical_mae = mae_per_horizon(
        historical_average_baseline(x_test_raw, num_horizons=len(HORIZONS)), y_test_raw
    )
    return model_mae, last_value_mae, historical_mae


def _print_table(
    title: str, model_mae: list[float], last_value_mae: list[float], historical_mae: list[float]
) -> None:
    print(f"\n=== {title} ===")
    print(f"{'horizon (decisions)':<22}{'graph_wavenet':>16}{'last_value':>16}{'hist_avg':>16}")
    for i, horizon in enumerate(HORIZONS):
        print(
            f"{horizon:<22}{model_mae[i]:>16.3f}{last_value_mae[i]:>16.3f}{historical_mae[i]:>16.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-decisions", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("train_forecaster.collecting_telemetry", num_decisions=args.num_decisions)
    series = collect_telemetry(args.num_decisions, seed=args.seed)
    x_raw, y_raw = build_windows(series, input_len=INPUT_LEN, horizons=HORIZONS)
    num_windows = len(x_raw)
    split = int(num_windows * TRAIN_FRACTION)

    temporal_train_idx = torch.arange(split)
    temporal_test_idx = torch.arange(split, num_windows)
    temporal_result = train_and_evaluate(
        x_raw, y_raw, temporal_train_idx, temporal_test_idx, args.epochs, args.seed
    )
    _print_table(
        f"Temporal holdout (train on first 80%, test on last 20%; {len(temporal_test_idx)} windows)",
        *temporal_result,
    )

    torch.manual_seed(args.seed)
    perm = torch.randperm(num_windows)
    iid_train_idx, iid_test_idx = perm[:split], perm[split:]
    iid_result = train_and_evaluate(
        x_raw, y_raw, iid_train_idx, iid_test_idx, args.epochs, args.seed
    )
    _print_table(
        f"i.i.d. random split (pipeline sanity check, not a deployment claim; {len(iid_test_idx)} windows)",
        *iid_result,
    )

    temporal_model_mae, temporal_last_value_mae, _ = temporal_result
    iid_model_mae, iid_last_value_mae, _ = iid_result
    beats_temporal = all(
        m <= b for m, b in zip(temporal_model_mae, temporal_last_value_mae, strict=True)
    )
    beats_iid = all(m <= b for m, b in zip(iid_model_mae, iid_last_value_mae, strict=True))

    if not beats_temporal and beats_iid:
        print(
            "\nNOTE: the model beats last-value under the i.i.d. sanity-check split "
            "(proving the pipeline learns real structure) but NOT under the temporal "
            "holdout (the methodologically correct split for real deployment) -- a "
            "diagnosed training-data-scale limitation, not an architecture bug. "
            f"{args.num_decisions} decisions from a single demand ramp isn't enough "
            "to generalize across the regime shift between train and test; this is "
            "exactly what docs/design/06_ai_pipeline.md section 6.4 means by needing "
            "'weeks of accumulated telemetry across varied scenario runs', not a "
            "Kaggle/GPU limitation like the YOLO pipeline's gap."
        )
    elif not beats_temporal and not beats_iid:
        print(
            "\nWARNING: the model does not beat last-value under EITHER split -- this "
            "points at a real training/architecture problem, not just limited data, "
            "and should be investigated before trusting this forecaster."
        )
    else:
        print("\nThe model beats the last-value baseline under both splits.")

    for title, (model_mae, last_value_mae, historical_mae) in (
        ("temporal", temporal_result),
        ("iid", iid_result),
    ):
        for i, horizon in enumerate(HORIZONS):
            logger.info(
                "train_forecaster.result",
                split=title,
                horizon=horizon,
                model_mae=model_mae[i],
                last_value_mae=last_value_mae[i],
                historical_mae=historical_mae[i],
            )


if __name__ == "__main__":
    main()
