"""Validates LstmAutoencoder's shape/gradient correctness with synthetic tensors, and
--the real question for a reconstruction-error detector -- that it actually learns to
reconstruct familiar patterns better than unfamiliar ones after training, not just that
it runs without error.
"""

from __future__ import annotations

import pytest
import torch

from atlas.anomaly.lstm_autoencoder import LstmAutoencoder, fit_threshold

NUM_FEATURES = 3
SEQ_LEN = 10
BATCH = 4


def _model() -> LstmAutoencoder:
    return LstmAutoencoder(num_features=NUM_FEATURES, hidden_dim=8, num_layers=1)


class TestForwardShape:
    def test_output_shape_matches_input(self) -> None:
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)

        out = model(x)

        assert out.shape == x.shape

    def test_rejects_wrong_feature_count(self) -> None:
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES + 1)

        with pytest.raises(ValueError, match="expected"):
            model(x)

    def test_gradients_flow_to_every_parameter(self) -> None:
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES)

        out = model(x)
        out.sum().backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"


class TestReconstructionErrorLearning:
    """The actual point of a reconstruction-error detector: after training on ONE
    pattern, a familiar window should reconstruct better than a genuinely different one.
    """

    def test_trained_model_reconstructs_familiar_pattern_better_than_novel_one(self) -> None:
        torch.manual_seed(0)
        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train only on a smooth sine-wave-like pattern -- the "incident-free
        # telemetry" this tier is meant to learn.
        t = torch.linspace(0, 4 * 3.14159, SEQ_LEN)
        normal_pattern = torch.stack([torch.sin(t + i) for i in range(NUM_FEATURES)], dim=-1)
        normal_batch = normal_pattern.unsqueeze(0).expand(BATCH, -1, -1)

        for _ in range(200):
            optimizer.zero_grad()
            reconstruction = model(normal_batch)
            loss = torch.nn.functional.mse_loss(reconstruction, normal_batch)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        model.eval()
        familiar_error = model.reconstruction_error(normal_pattern)

        # A genuinely different pattern: a sharp step function, nothing like a sine
        # wave -- the "incident signature" this tier should reconstruct poorly.
        novel_pattern = torch.zeros(SEQ_LEN, NUM_FEATURES)
        novel_pattern[SEQ_LEN // 2 :] = 5.0

        novel_error = model.reconstruction_error(novel_pattern)

        assert novel_error > familiar_error


class TestFitThreshold:
    def test_hand_computed_percentile(self) -> None:
        """A model that always reconstructs perfectly (error 0) except when told to
        add a fixed offset -- lets us hand-pick exactly which reconstruction errors
        fit_threshold sees, rather than depending on real training dynamics."""

        class _FakeModel:
            def __init__(self, errors: list[float]) -> None:
                self._errors = iter(errors)

            def reconstruction_error(self, window: torch.Tensor) -> float:
                return next(self._errors)

        # 10 windows with errors 1..10; 90th percentile (index = int(10*0.9)=9) -> errors[9]=10.0
        fake_model = _FakeModel([float(i) for i in range(1, 11)])
        windows = [torch.zeros(1, 1) for _ in range(10)]

        threshold = fit_threshold(fake_model, windows, percentile=90.0)  # type: ignore[arg-type]

        assert threshold == pytest.approx(10.0)

    def test_rejects_empty_windows(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="non-empty"):
            fit_threshold(model, [])

    def test_rejects_percentile_out_of_range(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="percentile"):
            fit_threshold(model, [torch.zeros(SEQ_LEN, NUM_FEATURES)], percentile=100.0)
