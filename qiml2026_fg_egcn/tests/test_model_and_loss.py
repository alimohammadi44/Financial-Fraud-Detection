import torch

from fg_egcn.losses import labeled_focal_loss
from fg_egcn.model import FGEGCN, normalized_directed_propagation


def test_normalized_propagation_shape_and_finite():
    x = torch.randn(5, 7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = normalized_directed_propagation(x, edge_index)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_fg_egcn_forward_exposes_evidence_streams():
    torch.manual_seed(1)
    x = torch.randn(12, 94)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        dtype=torch.long,
    )
    model = FGEGCN(input_dim=94, hidden_dim=64, num_layers=2, dropout=0.0, gate_hidden_dim=64)
    out = model.forward_snapshot(x, edge_index, model.initial_states())
    assert out["logits"].shape == (12, 2)
    assert out["z_graph"].shape == (12, 64)
    assert out["s_feature"].shape == (12, 64)
    assert out["gamma"].shape == (12, 1)
    assert out["fused"].shape == (12, 64)
    assert torch.all((out["gamma"] >= 0) & (out["gamma"] <= 1))
    reconstructed = out["s_feature"] + out["gamma"] * (out["z_graph"] - out["s_feature"])
    assert torch.allclose(out["fused"], reconstructed, atol=1e-6)


def test_focal_loss_ignores_unknown_nodes():
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [100.0, -100.0]], requires_grad=True)
    y = torch.tensor([0, 1, -1])
    loss, count = labeled_focal_loss(logits, y, gamma=2.0, alpha_licit=0.70, alpha_illicit=0.29)
    assert count == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad[2], torch.zeros(2))
