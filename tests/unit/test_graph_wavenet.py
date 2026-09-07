"""Validates GraphWaveNetForecaster's shapes/causality and quantile_loss's hand-computed
values -- no real telemetry needed, synthetic tensors are enough to verify the
architecture is actually correct (causal, right output shape, gradients flow).
"""

from __future__ import annotations

import pytest
import torch

from atlas.forecasting.graph_wavenet import GraphWaveNetForecaster, quantile_loss

NUM_NODES = 4
IN_FEATURES = 2
SEQ_LEN = 20
BATCH = 3
HORIZONS = (3, 6, 12)
QUANTILES = (0.1, 0.5, 0.9)


def _model() -> GraphWaveNetForecaster:
    return GraphWaveNetForecaster(
        num_nodes=NUM_NODES,
        in_features=IN_FEATURES,
        hidden_dim=8,
        num_temporal_layers=2,
        embed_dim=4,
        horizons=HORIZONS,
        quantiles=QUANTILES,
    )


class TestAdaptiveAdjacency:
    def test_shape_is_num_nodes_by_num_nodes(self) -> None:
        model = _model()
        adj = model.adaptive_adjacency()
        assert adj.shape == (NUM_NODES, NUM_NODES)

    def test_rows_sum_to_one(self) -> None:
        """A softmax over each node's neighbor weights -- a weighted-aggregation
        operator, verified directly rather than assumed from the softmax call alone."""
        model = _model()
        adj = model.adaptive_adjacency()
        row_sums = adj.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(NUM_NODES), atol=1e-5)

    def test_entries_are_non_negative(self) -> None:
        model = _model()
        adj = model.adaptive_adjacency()
        assert torch.all(adj >= 0)


class TestForwardShape:
    def test_output_shape_matches_batch_nodes_horizons_quantiles(self) -> None:
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_NODES, IN_FEATURES)

        out = model(x)

        assert out.shape == (BATCH, NUM_NODES, len(HORIZONS), len(QUANTILES))

    def test_rejects_wrong_node_count(self) -> None:
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_NODES + 1, IN_FEATURES)

        with pytest.raises(ValueError, match="expected"):
            model(x)

    def test_gradients_flow_to_every_parameter(self) -> None:
        """A basic sanity check that nothing in the forward pass detaches the graph --
        every parameter (including the node embeddings) must receive a gradient."""
        model = _model()
        x = torch.randn(BATCH, SEQ_LEN, NUM_NODES, IN_FEATURES)

        out = model(x)
        out.sum().backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"

    def test_last_timestep_is_within_the_models_receptive_field(self) -> None:
        """The most basic causality sanity check: the most recent input timestep must
        be able to influence the forecast at all. A model that ignored recent history
        (e.g. from an off-by-one in the causal padding swallowing real data) would fail
        this trivially while still returning a correctly-SHAPED output."""
        model = _model()
        torch.manual_seed(0)
        x = torch.randn(1, SEQ_LEN, NUM_NODES, IN_FEATURES)

        x_changed_last = x.clone()
        x_changed_last[0, -1] += 5.0

        with torch.no_grad():
            out_original = model(x)
            out_changed = model(x_changed_last)

        assert not torch.allclose(out_original, out_changed)


class TestQuantileLossHandComputed:
    """clip: pred=[1.0, 2.0, 3.0] for quantiles (0.1, 0.5, 0.9), target=4.0.
    errors = target - pred = [3.0, 2.0, 1.0]
    q=0.1: max((0.1-1)*3.0, 0.1*3.0) = max(-2.7, 0.3) = 0.3
    q=0.5: max((0.5-1)*2.0, 0.5*2.0) = max(-1.0, 1.0) = 1.0
    q=0.9: max((0.9-1)*1.0, 0.9*1.0) = max(-0.1, 0.9) = 0.9
    mean = (0.3 + 1.0 + 0.9) / 3 = 0.7333...
    """

    def test_hand_computed_value(self) -> None:
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([4.0])

        loss = quantile_loss(preds, target, quantiles=(0.1, 0.5, 0.9))

        assert loss.item() == pytest.approx(0.7333333, abs=1e-5)

    def test_p50_head_alone_is_half_of_mae(self) -> None:
        preds = torch.tensor([[5.0]])
        target = torch.tensor([2.0])

        loss = quantile_loss(preds, target, quantiles=(0.5,))

        assert loss.item() == pytest.approx(0.5 * abs(2.0 - 5.0))

    def test_zero_error_gives_zero_loss(self) -> None:
        preds = torch.tensor([[3.0, 3.0, 3.0]])
        target = torch.tensor([3.0])

        loss = quantile_loss(preds, target, quantiles=(0.1, 0.5, 0.9))

        assert loss.item() == pytest.approx(0.0)
