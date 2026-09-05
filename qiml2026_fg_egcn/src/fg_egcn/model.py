from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def normalized_directed_propagation(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Apply D^-1/2 (A+I) D^-1/2 to source-to-target messages.

    FG-EGCN describes the Elliptic edges as directed and applies symmetric GCN
    normalization after self-loop addition. We preserve the listed direction and
    use the target-degree convention used by common source-to-target sparse GCN
    implementations. This sparse orientation is recorded as a reproduction
    choice because the article does not publish its adjacency-construction code.
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


class MatrixGRUGate(nn.Module):
    """Matrix-GRU gate used by the original EvolveGCN-H formulation."""

    def __init__(self, rows: int, cols: int, activation: str) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(rows, rows))
        self.U = nn.Parameter(torch.empty(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, cols))
        bound = 1.0 / math.sqrt(rows)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        if activation not in {"sigmoid", "tanh"}:
            raise ValueError("activation must be sigmoid or tanh")
        self.activation = activation

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        value = self.W @ x + self.U @ hidden + self.bias
        return torch.sigmoid(value) if self.activation == "sigmoid" else torch.tanh(value)


class MatrixGRUCell(nn.Module):
    """Evolve a d_in x d_out graph-convolution weight matrix."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.update = MatrixGRUGate(rows, cols, "sigmoid")
        self.reset = MatrixGRUGate(rows, cols, "sigmoid")
        self.candidate = MatrixGRUGate(rows, cols, "tanh")

    def forward(self, summary: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        update = self.update(summary, previous)
        reset = self.reset(summary, previous)
        candidate = self.candidate(summary, reset * previous)
        return (1.0 - update) * previous + update * candidate


class EvolveWeightGCNLayer(nn.Module):
    """FG-EGCN temporal graph layer with EvolveGCN-H matrix weight evolution.

    Han et al. specify k=d_l TopK rows and a linear projection when the summary
    shape does not match W_t. The original EvolveGCN-H summarizer normalizes its
    scoring vector and weights selected rows by tanh(score). We combine these
    two explicit specifications: select k=d_in nodes, score-weight them, transpose
    to d_in x d_in, then project to d_in x d_out before the matrix-GRU update.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout

        self.initial_weight = nn.Parameter(torch.empty(d_in, d_out))
        bound = 1.0 / math.sqrt(d_out)
        nn.init.uniform_(self.initial_weight, -bound, bound)

        self.score_vector = nn.Parameter(torch.empty(d_in, 1))
        score_bound = 1.0 / math.sqrt(d_in)
        nn.init.uniform_(self.score_vector, -score_bound, score_bound)

        # FG-EGCN Eq. (7) explicitly allows projection to align the TopK
        # summary with W_t. For d_in != d_out this maps columns accordingly.
        self.summary_projection = nn.Linear(d_in, d_out)
        self.weight_gru = MatrixGRUCell(rows=d_in, cols=d_out)

        self.residual_projection = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out)

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((self.d_in, self.d_out))

        norm = self.score_vector.norm().clamp_min(1e-12)
        scores = (x @ self.score_vector / norm).reshape(-1)
        k = min(self.d_in, x.shape[0])
        values, idx = torch.topk(scores, k=k, largest=True, sorted=False)
        selected = x[idx] * torch.tanh(values).unsqueeze(-1)

        # Real Elliptic snapshots contain far more than d_in nodes. Padding is
        # only a defensive path for tiny synthetic/unit-test snapshots.
        if k < self.d_in:
            if k == 0:
                selected = x.new_zeros((self.d_in, self.d_in))
            else:
                selected = torch.cat(
                    [selected, selected[-1:].expand(self.d_in - k, -1)], dim=0
                )

        # selected: d_in x d_in; original EvolveGCN-H TopK transposes the
        # selected-node matrix before passing it to the matrix GRU.
        summary_square = selected.transpose(0, 1)
        return self.summary_projection(summary_square)

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
        dims = [input_dim] + [hidden_dim] * num_layers
        self.temporal_layers = nn.ModuleList(
            [EvolveWeightGCNLayer(dims[i], dims[i + 1], dropout) for i in range(num_layers)]
        )

        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        # The paper denotes this nonlinearity as delta without naming it; ReLU
        # is an explicit reproduction assumption and is recorded in the notes.
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

        return {
            "logits": logits,
            "z_graph": z,
            "s_feature": s,
            "gamma": gamma,
            "fused": fused,
            "states": new_states,
        }
