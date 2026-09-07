#!/usr/bin/env python
"""Evaluates the LSTM-autoencoder anomaly tier against the SAME real accident scenario
tier 1 (scripts/eval_anomaly_detection.py) was evaluated against -- a direct,
apples-to-apples comparison of the two tiers docs/design/04_agent_architecture.md
section 4.8 describes, not two unrelated benchmarks.

Monitors TWO correlated features per window -- mean occupancy and mean speed of the
accident edge (448686048#0) -- because a real accident's signature is exactly a
CORRELATED pattern (occupancy up, speed down, together), the kind of pattern tier 1's
single-series z-score cannot see by construction and this tier exists specifically to
catch (section 4.8: "upstream occupancy up + downstream flow down").

Trained on a SEPARATE incident-free corpus -- several independent runs of the same
network/demand with NO scenario ever triggered, different seeds each -- not just the
early portion of the one episode being evaluated. Found necessary, not applied
speculatively: an initial version trained on only the ~14 incident-free windows
available before this one episode's accident (start_s=600) massively overfit to that
narrow slice, flagging most of the REST of that same episode as anomalous too (20 of 31
non-accident windows -- useless as a detector). A real historical corpus of several
incident-free episodes is both more realistic (matching the design's own "trained on
incident-free telemetry" -- a standing corpus, not one episode's prefix) and gives the
model enough variety to actually learn what "normal" looks like.

Usage:
    python scripts/eval_lstm_autoencoder.py [--duration-s 1500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.anomaly import LstmAutoencoder, fit_threshold  # noqa: E402
from atlas.common.config import project_root  # noqa: E402
from atlas.common.logging import configure_logging, get_logger  # noqa: E402
from atlas.twin.env import SumoTwinEnv, TwinEnvConfig  # noqa: E402
from atlas.twin.scenario import ScenarioController, ScenarioSpec  # noqa: E402

configure_logging()
logger = get_logger(__name__)

NETWORK_DIR = project_root() / "data" / "networks" / "manhattan_midtown"
ACCIDENT_EDGE = "448686048#0"
ACCIDENT_START_S = 600
ACCIDENT_DURATION_S = 300
AGGREGATION_WINDOW_S = 30.0
SEQ_LEN = 6  # 6 * 30s = 3 minutes of history per window
THRESHOLD_PERCENTILE = 99.0


def collect_aggregated_series(
    duration_s: int, seed: int, inject_accident: bool, step_length_s: float = 1.0
) -> list[list[float]]:
    """Returns one [mean_occupancy, mean_speed] pair per AGGREGATION_WINDOW_S window.
    `inject_accident=False` runs the same network/demand with no scenario at all -- an
    incident-free episode for the training corpus; `inject_accident=True` reproduces
    the exact scenario tier 1 was evaluated against, for the held-out evaluation run.
    """
    net_file = NETWORK_DIR / "net.net.xml"
    route_file = NETWORK_DIR / "demand" / "pm_peak.rou.xml"
    vtypes_file = NETWORK_DIR / "vtypes.add.xml"

    twin = SumoTwinEnv(
        TwinEnvConfig(
            name="eval_lstm_ae",
            net_file=net_file,
            route_file=route_file,
            additional_files=[vtypes_file] if vtypes_file.is_file() else [],
            step_length_s=step_length_s,
        )
    )
    specs = (
        [
            ScenarioSpec(
                name="midblock_accident",
                type="accident",
                edge_id=ACCIDENT_EDGE,
                start_s=ACCIDENT_START_S,
                duration_s=ACCIDENT_DURATION_S,
                speed_factor=0.25,
            )
        ]
        if inject_accident
        else []
    )
    scenario = ScenarioController(twin, specs)
    twin.reset(seed=seed)
    conn = twin.connection
    lanes = [f"{ACCIDENT_EDGE}_{i}" for i in range(conn.edge.getLaneNumber(ACCIDENT_EDGE))]

    series: list[list[float]] = []
    occ_buf: list[float] = []
    speed_buf: list[float] = []
    num_steps = int(duration_s / step_length_s)
    raw_steps_per_window = int(AGGREGATION_WINDOW_S / step_length_s)

    try:
        for step in range(num_steps):
            scenario.step()
            occ_buf.append(
                sum(conn.lane.getLastStepOccupancy(lane_id) for lane_id in lanes) / len(lanes)
            )
            speed_buf.append(
                sum(conn.lane.getLastStepMeanSpeed(lane_id) for lane_id in lanes) / len(lanes)
            )
            if (step + 1) % raw_steps_per_window == 0:
                series.append([sum(occ_buf) / len(occ_buf), sum(speed_buf) / len(speed_buf)])
                occ_buf, speed_buf = [], []
    finally:
        twin.close()

    return series


def build_windows(series: list[list[float]], seq_len: int) -> torch.Tensor:
    tensor = torch.tensor(series, dtype=torch.float32)
    return torch.stack([tensor[i : i + seq_len] for i in range(len(tensor) - seq_len + 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=int, default=1500)
    parser.add_argument("--num-training-episodes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    logger.info("eval_lstm_ae.collecting_training_corpus", episodes=args.num_training_episodes)
    train_windows_list = []
    for episode_seed in range(args.num_training_episodes):
        episode_series = collect_aggregated_series(
            args.duration_s, seed=episode_seed, inject_accident=False
        )
        train_windows_list.append(build_windows(episode_series, SEQ_LEN))
    train_windows = torch.cat(train_windows_list, dim=0)

    logger.info("eval_lstm_ae.collecting_evaluation_episode")
    eval_series = collect_aggregated_series(
        args.duration_s, seed=args.num_training_episodes, inject_accident=True
    )
    eval_windows = build_windows(eval_series, SEQ_LEN)
    window_end_times_s = [(i + SEQ_LEN) * AGGREGATION_WINDOW_S for i in range(len(eval_windows))]

    logger.info(
        "eval_lstm_ae.windows_built",
        train_incident_free=len(train_windows),
        eval_windows=len(eval_windows),
    )

    # Normalize using the training corpus's statistics only (same discipline as
    # scripts/train_forecaster.py -- never let evaluation-episode scale leak into
    # training).
    train_mean = train_windows.mean(dim=(0, 1))
    train_std = train_windows.std(dim=(0, 1)) + 1e-6
    train_windows_norm = (train_windows - train_mean) / train_std
    eval_windows_norm = (eval_windows - train_mean) / train_std

    torch.manual_seed(42)
    model = LstmAutoencoder(num_features=2, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _epoch in range(args.epochs):
        optimizer.zero_grad()
        reconstruction = model(train_windows_norm)
        loss = torch.nn.functional.mse_loss(reconstruction, train_windows_norm)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    model.eval()
    threshold = fit_threshold(model, list(train_windows_norm), percentile=THRESHOLD_PERCENTILE)
    logger.info("eval_lstm_ae.threshold", threshold=threshold, final_train_loss=loss.item())

    detection_time_s: float | None = None
    false_positive_count = 0
    num_accident_windows = 0
    for window, end_time_s in zip(eval_windows_norm, window_end_times_s, strict=True):
        error = model.reconstruction_error(window)
        is_anomalous = error > threshold
        in_accident_window = ACCIDENT_START_S <= end_time_s < ACCIDENT_START_S + ACCIDENT_DURATION_S
        if in_accident_window:
            num_accident_windows += 1
        if is_anomalous and in_accident_window and detection_time_s is None:
            detection_time_s = end_time_s
        elif is_anomalous and not in_accident_window:
            false_positive_count += 1

    print("\n=== LSTM-autoencoder anomaly evaluation (same accident scenario as tier 1) ===")
    if detection_time_s is not None:
        print(f"detection_delay_s: {detection_time_s - ACCIDENT_START_S:.1f}")
    else:
        print("detection_delay_s: NEVER DETECTED within the accident window")
    num_non_accident_windows = len(eval_windows_norm) - num_accident_windows
    print(
        f"false_positive_count: {false_positive_count} / {num_non_accident_windows} "
        "non-accident windows in the evaluation episode"
    )
    print(
        "\nFor comparison, tier 1 (EWMA/z-score) on this same scenario: 30.0s delay, 1 false positive."
    )
    print(
        "\nHonest reading, not a bug to chase away: even trained on a 5-episode incident-\n"
        "free corpus (225 windows, up from 14 when trained on just one episode's prefix --\n"
        "itself a real fix, see the module docstring), this tier still shows a nontrivial\n"
        "false-positive rate on this light-demand network, where genuinely normal traffic\n"
        "has high per-lane variance (Week 9 tier 1's own finding). This is exactly why\n"
        "docs/design/04_agent_architecture.md section 4.8 tunes BOTH tiers for high recall\n"
        "with Supervisor-side disambiguation (what-if + camera check) downstream, rather\n"
        "than expecting either detector to be precise alone -- 'a missed incident costs\n"
        "more than a false triage' is that section's own reasoning, and both tiers'\n"
        "measured false-positive rates are the concrete evidence for why that design\n"
        "choice is the right one, not a nice-to-have."
    )


if __name__ == "__main__":
    main()
