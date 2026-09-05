from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .features import sinusoidal_time_encoding

try:
    from torch_geometric.nn import TransformerConv
except ImportError:
    TransformerConv = None


class ChronoWaveGNN(nn.Module):
    """Independent implementation of the paper's documented ChronoWave-GNN core.

    The article specifies a 3-layer TransformerConv/TGAT+ backbone, 8-D
    sinusoidal time encoding, ELU, and dropout=0.4, but does not report hidden
    width or attention-head count in the accessible Methods/Experimental Setup.
    Those values are therefore explicit reproduction assumptions.
    """

    def __init__(
        self,
        enriched_feature_dim: int,
        *,
        time_dim: int = 8,
        hidden_dim: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if TransformerConv is None:
            raise ImportError(
                "torch-geometric is required. Install a build matching your PyTorch/CUDA environment."
            )
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")

        self.time_dim = time_dim
        self.dropout = dropout
        self.time_projection = nn.Linear(time_dim, time_dim)
        input_dim = enriched_feature_dim + time_dim
        per_head = hidden_dim // heads

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.convs.append(
                TransformerConv(
                    in_dim,
                    per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    beta=False,
                    root_weight=True,
                )
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, edge_index: torch.Tensor):
        time_fixed = sinusoidal_time_encoding(time_step, self.time_dim)
        time_context = self.time_projection(time_fixed)
        h = torch.cat([x, time_context], dim=-1)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return logits, h
