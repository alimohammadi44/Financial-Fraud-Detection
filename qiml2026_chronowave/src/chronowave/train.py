from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import GraphBundle
from .metrics import classification_metrics
from .model import ChronoWaveGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probabilities(model, bundle: GraphBundle, edge_index: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(bundle.x.to(device), bundle.time_step.to(device), edge_index.to(device))
        prob = torch.softmax(logits, dim=-1)[:, 1]
    return prob.cpu().numpy(), embeddings.cpu()


def train_one_seed(bundle: GraphBundle, cfg: Dict, seed: int, out_dir: Path) -> dict:
    set_seed(seed)
    requested = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model = ChronoWaveGNN(
        enriched_feature_dim=bundle.x.shape[1],
        time_dim=model_cfg.get("time_dim", 8),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        heads=model_cfg.get("heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.4),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.005)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("eta_min", 0.0)),
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1))
    )

    x = bundle.x.to(device)
    ts = bundle.time_step.to(device)
    y = bundle.y.to(device)
    train_mask = bundle.train_mask.to(device)
    val_mask_np = bundle.val_mask.numpy()
    y_np = bundle.y.numpy()

    # The article says early stopping uses validation F1 but does not specify
    # the averaging convention. Make it explicit/configurable rather than
    # silently assuming illicit-class, macro, or weighted F1.
    validation_metric = str(train_cfg.get("validation_metric", "weighted_f1"))
    allowed = {"illicit_f1", "macro_f1", "weighted_f1", "pr_auc", "roc_auc"}
    if validation_metric not in allowed:
        raise ValueError(f"validation_metric must be one of {sorted(allowed)}")

    best_score = -float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    patience = int(train_cfg.get("patience", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, ts, bundle.train_edge_index.to(device))
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        val_prob, _ = _probabilities(model, bundle, bundle.val_edge_index, device)
        val_metrics = classification_metrics(y_np[val_mask_np], val_prob[val_mask_np])
        score = float(val_metrics[validation_metric])

        if score > best_score + 1e-12:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    test_prob, embeddings = _probabilities(model, bundle, bundle.test_edge_index, device)
    test_mask = bundle.test_mask.numpy()
    metrics = classification_metrics(y_np[test_mask], test_prob[test_mask])
    metrics.update({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metric": validation_metric,
        "best_validation_score": best_score,
    })

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "best_model.pt")
    torch.save(embeddings, seed_dir / "node_embeddings.pt")
    np.save(seed_dir / "test_prob_illicit.npy", test_prob)
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def summarize(results: list[dict]) -> dict:
    numeric = [
        key for key, value in results[0].items()
        if isinstance(value, (int, float)) and key != "seed"
    ]
    summary = {"n_seeds": len(results), "seeds": [r["seed"] for r in results]}
    for key in numeric:
        arr = np.asarray([r[key] for r in results], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    summary["validation_metric"] = results[0].get("validation_metric")
    return summary
