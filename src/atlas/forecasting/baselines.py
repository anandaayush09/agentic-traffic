"""The two forecasting baselines docs/design/06_ai_pipeline.md section 6.4 evaluates
against: "MAE/RMSE per horizon vs. historical-average and last-value baselines". A
learned forecaster earning its complexity has to actually beat these -- both are
zero-training, and last-value in particular is a genuinely strong baseline for
short-horizon traffic state (link speed rarely swings wildly in 15 minutes).
"""

from __future__ import annotations

from torch import Tensor


def last_value_baseline(x_windows: Tensor, num_horizons: int) -> Tensor:
    """Predicts the input window's most recent feature-0 value, unchanged, at every
    horizon. `x_windows`: `(num_windows, input_len, num_nodes, in_features)`. Returns
    `(num_windows, num_nodes, num_horizons)`.
    """
    last = x_windows[:, -1, :, 0]  # (num_windows, num_nodes)
    return last.unsqueeze(-1).expand(-1, -1, num_horizons)


def historical_average_baseline(x_windows: Tensor, num_horizons: int) -> Tensor:
    """Predicts the input window's mean feature-0 value, at every horizon."""
    mean = x_windows[..., 0].mean(dim=1)  # (num_windows, num_nodes)
    return mean.unsqueeze(-1).expand(-1, -1, num_horizons)
