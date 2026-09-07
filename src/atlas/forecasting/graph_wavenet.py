"""A Graph WaveNet-class spatio-temporal forecaster (docs/design/06_ai_pipeline.md
section 6.4: "graph convolutions with learned adaptive adjacency + dilated temporal
convolutions... quantile heads (p10/p50/p90) for honest uncertainty"). Scaled down from
the design's ~300K-param target (this project's actual graph is 5 corridor junctions,
not a city-scale sensor network) but the same architectural ideas, each independently
verifiable: an adaptive adjacency learned from node embeddings rather than a fixed graph,
a per-node dilated causal temporal convolution stack (weight-shared across nodes, the
same convolution kernels see every node's series), and a final graph-mixing step before
the quantile output heads.

Causal padding (`(kernel_size - 1) * dilation` zeros prepended, never appended) is what
makes the temporal convolutions valid forecasting inputs at all -- without it, a
"dilated conv" would leak future timesteps into its own receptive field, exactly the
kind of subtle bug that produces suspiciously-good backtest numbers and a model that's
useless in an actual forecasting loop. Verified directly in tests/unit/test_graph_
wavenet.py by checking output length before slicing, not just trusting the formula.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812 (the universal PyTorch convention)
from torch import Tensor, nn

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


class GraphWaveNetForecaster(nn.Module):
    """Forecasts `len(quantiles)` quantiles at each of `horizons` future steps, for
    every node, from a length-`seq_len` history window.

    Input: `(batch, seq_len, num_nodes, in_features)`.
    Output: `(batch, num_nodes, len(horizons), len(quantiles))`.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int = 32,
        num_temporal_layers: int = 3,
        embed_dim: int = 8,
        horizons: tuple[int, ...] = (3, 6, 12),
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        if num_nodes < 1 or in_features < 1:
            raise ValueError("num_nodes and in_features must both be >= 1")
        self.num_nodes = num_nodes
        self.horizons = horizons
        self.quantiles = quantiles

        # Learned node embeddings -> an adaptive adjacency (Graph WaveNet's own
        # "self-adaptive adjacency matrix" idea), rather than a hand-specified graph.
        self.node_embed_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_embed_2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.temporal_layers = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i)
                for i in range(num_temporal_layers)
            ]
        )
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, len(horizons) * len(quantiles))

    def adaptive_adjacency(self) -> Tensor:
        """(num_nodes, num_nodes), each row a softmax over that node's learned
        similarity to every other node -- rows sum to 1, so this is a weighted
        neighbor-aggregation operator, not a raw similarity score.
        """
        logits = F.relu(self.node_embed_1 @ self.node_embed_2.T)
        return F.softmax(logits, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, num_nodes, _ = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {num_nodes}")

        h = self.input_proj(x)  # (batch, seq_len, num_nodes, hidden)
        h = h.permute(0, 2, 3, 1).reshape(batch * num_nodes, -1, seq_len)  # (B*N, hidden, T)

        for module in self.temporal_layers:
            conv = cast(nn.Conv1d, module)
            pad_left = conv.dilation[0] * (conv.kernel_size[0] - 1)
            h = F.pad(h, (pad_left, 0))  # causal: pad the past, never the future
            h = torch.tanh(conv(h))

        h_last = h[:, :, -1].reshape(batch, num_nodes, -1)  # (batch, nodes, hidden)

        adjacency = self.adaptive_adjacency()
        h_graph = torch.einsum("nm,bmc->bnc", adjacency, h_last)
        h_graph = F.relu(self.graph_proj(h_graph)) + h_last  # residual

        out = self.output_proj(h_graph)  # (batch, nodes, horizons*quantiles)
        return cast(Tensor, out.view(batch, num_nodes, len(self.horizons), len(self.quantiles)))


def quantile_loss(
    preds: Tensor, target: Tensor, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
) -> Tensor:
    """Pinball loss, averaged over every quantile head. `preds`: (..., len(quantiles));
    `target`: (...) (broadcasts against each quantile). Verified by hand (see
    tests/unit/test_graph_wavenet.py): asymmetric by design -- for the p10 head,
    UNDER-shooting the target is cheap (that's what a low quantile is supposed to do)
    and OVER-shooting it is expensive, and vice versa for p90; p50 collapses to
    ordinary MAE / 2.
    """
    target_expanded = target.unsqueeze(-1)
    errors = target_expanded - preds
    quantile_tensor = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)
    losses = torch.maximum((quantile_tensor - 1) * errors, quantile_tensor * errors)
    return losses.mean()
