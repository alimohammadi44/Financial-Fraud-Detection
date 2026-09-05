from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def normalized_directed_propagation(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Apply D^-1/2 (A+I) D^-1/2 to source-to-target messages.

    The paper retains directed transaction edges and specifies symmetric GCN
    normalization after adding self-loops. For sparse edge-index execution we
    treat each listed edge as source->target and use the target-degree vector in
    the same convention as common source-to-target GCN implementations.
    """
    n = x.shape[0]
    device = x.device
    loops = torch.arange(n, device=device, dtype=torch.long)
    if edge_index.numel() == 0:
        src = loops
        dst = loops
    else:
        src = torch.cat([edge_index[0].to(device), loops], dim=0)
        dst = torch.cat([edge_index[1].to(device), loops], dim=0)

    weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
    deg = torch.zeros(n, device=device, dtype=x.dtype)
    deg.index_add_(0, dst, weight)
    inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    norm = inv_sqrt[src] * weight * inv_sqrt[dst]

    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] * norm.unsqueeze(-1))
    return out


class EvolveWeightGCNLayer(nn.Module):
    """Equation-level EvolveGCN-H-style graph layer from the FG-EGCN paper.

    W_t is the recurrent hidden state. A learnable projection vector ranks nodes;
    the top d_in rows form a compact summary, which is linearly aligned to the
    d_in x d_out weight-state shape before a row-wise GRU update.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout

        self.initial_weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.xavier_uniform_(self.initial_weight)
        self.score_vector = nn.Parameter(torch.empty(d_in))
        nn.init.normal_(self.score_vector, mean=0.0, std=1.0 / max(1, d_in) ** 0.5)

        # Eq. (7) says a linear projection is used when TopK summary width and
        # W_t do not match. This maps each selected d_in-dimensional row to d_out.
        self.summary_projection = nn.Linear(d_in, d_out)
        self.weight_gru = nn.GRUCell(input_size=d_out, hidden_size=d_out)

        self.residual_projection = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out)

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        scores = x @ self.score_vector
        k = min(self.d_in, x.shape[0])
        if k == 0:
            return x.new_zeros((self.d_in, self.d_out))
        idx = torch.topk(scores, k=k, largest=True, sorted=False).indices
        selected = x[idx]
        if k < self.d_in:
            selected = torch.cat(
                [selected, x.new_zeros((self.d_in - k, self.d_in))], dim=0
            )
        return self.summary_projection(selected)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        previous_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        summary = self._summary(x)
        evolved_weight = self.weight_gru(summary, previous_weight)

        support = x @ evolved_weight
        propagated = normalized_directed_propagation(support, edge_index)
        propagated = F.relu(propagated)
        propagated = F.dropout(propagated, p=self.dropout, training=self.training)

        residual = self.residual_projection(x)
        out = self.layer_norm(propagated + residual)
        return out, evolved_weight


class FGEGCN(nn.Module):
    """Feature-gated temporal graph model following Han et al. (2026)."""

    def __init__(
        self,
        input_dim: int = 94,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        gate_hidden_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            # The class supports other values, but the published configuration is 2.
            pass
        dims = [input_dim] + [hidden_dim] * num_layers
        self.temporal_layers = nn.ModuleList(
            [EvolveWeightGCNLayer(dims[i], dims[i + 1], dropout) for i in range(num_layers)]
        )

        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        # The paper denotes the feature/gate nonlinearity as delta but does not
        # name it. ReLU is the explicit initial reproduction assumption.
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def initial_states(self) -> List[torch.Tensor]:
        return [layer.initial_weight for layer in self.temporal_layers]

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        states: List[torch.Tensor],
    ):
        if len(states) != len(self.temporal_layers):
            raise ValueError("Temporal state list does not match number of graph layers")

        h = x
        new_states: List[torch.Tensor] = []
        for layer, state in zip(self.temporal_layers, states):
            h, new_state = layer(h, edge_index, state)
            new_states.append(new_state)
        z = h

        s = F.relu(self.feature_projection(x))
        gamma = torch.sigmoid(self.gate_mlp(torch.cat([z, s], dim=-1)))
        fused = s + gamma * (z - s)
        logits = self.classifier(fused)

        # Exposing z/s/gamma/fused is intentional: after the classical baseline
        # is frozen these tensors become the inputs to QIML routing analyses.
        return {
            "logits": logits,
            "z_graph": z,
            "s_feature": s,
            "gamma": gamma,
            "fused": fused,
            "states": new_states,
        }
