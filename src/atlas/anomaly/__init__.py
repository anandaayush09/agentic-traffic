"""Two-tier anomaly detection (docs/design/04_agent_architecture.md section 4.8):
EWMA/z-score (explainable, zero-training) and an LSTM-autoencoder tier for subtler
multivariate incident signatures.
"""

from atlas.anomaly.ewma import AnomalyEvent, AnomalyKind, EwmaAnomalyDetector
from atlas.anomaly.lstm_autoencoder import LstmAutoencoder, fit_threshold

__all__ = [
    "AnomalyEvent",
    "AnomalyKind",
    "EwmaAnomalyDetector",
    "LstmAutoencoder",
    "fit_threshold",
]
