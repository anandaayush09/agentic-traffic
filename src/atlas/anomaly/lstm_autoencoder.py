"""LSTM-autoencoder anomaly detection -- tier 2 of the two-tier detector (docs/design/
04_agent_architecture.md section 4.8: "an LSTM-autoencoder over multivariate per-
junction windows for subtler patterns (incident signatures: upstream occupancy up +
downstream flow down), trained on incident-free telemetry"). Where tier 1
(atlas.anomaly.ewma) tracks one series at a time, this tier's whole point is
MULTIVARIATE: reconstruction error over several correlated features together catches
patterns no single feature's own z-score would flag (that gap is exactly what Week 9's
tier-1 evaluation -- scripts/eval_anomaly_detection.py -- found and documented on this
project's own light-demand network).

Reconstruction-error anomaly detection, not classification: the model is trained ONLY
on incident-free windows to reconstruct normal multivariate patterns well; a window the
model reconstructs POORLY is the anomaly signal, with no labeled incident examples
needed for training (only for evaluating detection delay/precision, via the scenario
engine's own ground truth -- see scripts/eval_lstm_autoencoder.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 (the universal PyTorch convention)
from torch import Tensor, nn


class LstmAutoencoder(nn.Module):
    """Encodes a `(batch, seq_len, num_features)` window to a fixed-size latent vector
    (the encoder LSTM's final hidden state) and decodes it back to the same shape.
    Reconstruction error (`reconstruction_error`) is the anomaly score -- a window the
    model has never seen a pattern like before reconstructs poorly.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, num_features)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, num_features = x.shape
        if num_features != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {num_features}")

        _, (h_n, _c_n) = self.encoder(x)
        latent = h_n[-1].unsqueeze(1).expand(batch, seq_len, self.hidden_dim)
        decoded, _ = self.decoder(latent)
        return self.output_proj(decoded)  # type: ignore[no-any-return]

    def reconstruction_error(self, window: Tensor) -> float:
        """`window`: `(seq_len, num_features)`, a single window (no batch dim). Returns
        the mean-squared reconstruction error -- the anomaly score.
        """
        with torch.no_grad():
            reconstruction = self.forward(window.unsqueeze(0))
            return F.mse_loss(reconstruction, window.unsqueeze(0)).item()


def fit_threshold(
    model: LstmAutoencoder, incident_free_windows: Sequence[Tensor], percentile: float = 99.0
) -> float:
    """The threshold above which a reconstruction error counts as anomalous: the
    `percentile`-th percentile of reconstruction error over a set of windows KNOWN to
    be incident-free (docs/design/04_agent_architecture.md section 4.8's "trained on
    incident-free telemetry" -- this is that same idea applied to threshold-setting,
    not just weight-training). A percentile, not a fixed multiple of the mean, so the
    threshold adapts to however noisy normal reconstruction error actually is instead
    of assuming a particular distribution shape.
    """
    if not incident_free_windows:
        raise ValueError("incident_free_windows must be non-empty")
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")

    errors = sorted(model.reconstruction_error(window) for window in incident_free_windows)
    index = min(int(len(errors) * percentile / 100.0), len(errors) - 1)
    return errors[index]
