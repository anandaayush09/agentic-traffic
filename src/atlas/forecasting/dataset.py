"""Turns one long multi-node time series into (history window, future targets) pairs
for `GraphWaveNetForecaster` training -- a pure, SUMO-agnostic windowing function so it's
testable against a synthetic series (docs/design/06_ai_pipeline.md section 6.4's data
source is real simulated telemetry, but the windowing logic itself doesn't care where
the series came from).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_windows(
    series: Tensor, input_len: int, horizons: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    """`series`: `(total_steps, num_nodes, in_features)`. Returns
    `(X, Y)` where `X` is `(num_windows, input_len, num_nodes, in_features)` and `Y` is
    `(num_windows, num_nodes, len(horizons))` -- `Y[:, :, i]` is the series value
    `horizons[i]` steps after the end of the corresponding `X` window, for the series'
    first feature only (multi-feature targets aren't needed by this project's forecast
    target, link mean speed).

    A window starting at index `t` needs history `series[t : t+input_len]` and targets
    at `t+input_len-1+h` for each horizon `h` -- so the last usable start index is
    `total_steps - input_len - max(horizons)`.
    """
    total_steps, _num_nodes, _in_features = series.shape
    max_horizon = max(horizons)
    num_windows = total_steps - input_len - max_horizon + 1
    if num_windows < 1:
        raise ValueError(
            f"series too short: need at least {input_len + max_horizon} steps, got {total_steps}"
        )

    x_windows = []
    y_windows = []
    for start in range(num_windows):
        end = start + input_len
        x_windows.append(series[start:end])
        targets = torch.stack(
            [series[end - 1 + h, :, 0] for h in horizons], dim=-1
        )  # (num_nodes, len(horizons))
        y_windows.append(targets)

    return torch.stack(x_windows), torch.stack(y_windows)
