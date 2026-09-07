"""Spatio-temporal traffic forecasting (docs/design/06_ai_pipeline.md section 6.4): a
Graph WaveNet-class model with an adaptive adjacency and quantile heads, trained on the
digital twin's own simulated telemetry (`scripts/train_forecaster.py`).
"""

from atlas.forecasting.baselines import historical_average_baseline, last_value_baseline
from atlas.forecasting.dataset import build_windows
from atlas.forecasting.graph_wavenet import DEFAULT_QUANTILES, GraphWaveNetForecaster, quantile_loss

__all__ = [
    "GraphWaveNetForecaster",
    "quantile_loss",
    "DEFAULT_QUANTILES",
    "build_windows",
    "last_value_baseline",
    "historical_average_baseline",
]
