"""Hand-verified windowing: build_windows against a small, fully-traceable synthetic
series -- every X/Y pair checked against exactly what it should contain, not just shapes.
"""

from __future__ import annotations

import pytest
import torch

from atlas.forecasting.baselines import historical_average_baseline, last_value_baseline
from atlas.forecasting.dataset import build_windows


def _series(total_steps: int, num_nodes: int = 1) -> torch.Tensor:
    """A single-feature series where value at step t is just t -- makes every window's
    expected content obvious by inspection (node dimension is a constant repeat)."""
    values = torch.arange(total_steps, dtype=torch.float32)
    return values.view(total_steps, 1, 1).expand(total_steps, num_nodes, 1).clone()


class TestBuildWindows:
    def test_hand_traced_single_window(self) -> None:
        # series = [0,1,2,...,9], input_len=3, horizons=(1,2)
        # only window: X = series[0:3] = [0,1,2]; end index = 2 (0-indexed last of X)
        # Y[h=1] = series[2+1] = 3; Y[h=2] = series[2+2] = 4
        series = _series(6)
        x, y = build_windows(series, input_len=3, horizons=(1, 2))

        assert x.shape == (2, 3, 1, 1)  # num_windows = 6-3-2+1 = 2
        assert x[0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
        assert y[0, 0, :].tolist() == [3.0, 4.0]

        # second window: X = series[1:4] = [1,2,3]; end index=3; Y[h=1]=4, Y[h=2]=5
        assert x[1, :, 0, 0].tolist() == [1.0, 2.0, 3.0]
        assert y[1, 0, :].tolist() == [4.0, 5.0]

    def test_num_windows_matches_formula(self) -> None:
        series = _series(20)
        x, _ = build_windows(series, input_len=5, horizons=(3, 6, 12))
        assert x.shape[0] == 20 - 5 - 12 + 1

    def test_multiple_nodes_windowed_independently(self) -> None:
        series = torch.zeros(10, 2, 1)
        series[:, 0, 0] = torch.arange(10)
        series[:, 1, 0] = torch.arange(10) * 10

        _, y = build_windows(series, input_len=3, horizons=(1,))

        assert y[0, 0, 0] == 3.0  # node 0: series[2+1]
        assert y[0, 1, 0] == 30.0  # node 1: series[2+1]*10

    def test_rejects_series_too_short_for_the_requested_window(self) -> None:
        series = _series(4)
        with pytest.raises(ValueError, match="too short"):
            build_windows(series, input_len=3, horizons=(5,))


class TestBaselines:
    def test_last_value_baseline_repeats_the_final_input_step(self) -> None:
        series = _series(6)
        x, _ = build_windows(series, input_len=3, horizons=(1, 2))

        pred = last_value_baseline(x, num_horizons=2)

        assert pred[0, 0, :].tolist() == [2.0, 2.0]  # last of window 0 = series[2] = 2

    def test_historical_average_baseline_is_the_window_mean(self) -> None:
        series = _series(6)
        x, _ = build_windows(series, input_len=3, horizons=(1, 2))

        pred = historical_average_baseline(x, num_horizons=2)

        assert pred[0, 0, :].tolist() == pytest.approx([1.0, 1.0])  # mean(0,1,2) = 1.0
