import torch

from chronowave.model import ChronoWaveGNN


def test_transformer_forward_shape():
    # Tiny directed graph; verifies that the PyG TransformerConv stack and time
    # encoding integrate correctly before licensed Elliptic data are introduced.
    x = torch.randn(8, 12)
    t = torch.arange(1, 9)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0, 2],
         [1, 2, 3, 4, 5, 6, 7, 4, 6]],
        dtype=torch.long,
    )
    model = ChronoWaveGNN(
        enriched_feature_dim=12,
        time_dim=8,
        hidden_dim=32,
        heads=4,
        num_layers=3,
        dropout=0.0,
    )
    logits, embeddings = model(x, t, edge_index)
    assert logits.shape == (8, 2)
    assert embeddings.shape == (8, 32)
    assert torch.isfinite(logits).all()
