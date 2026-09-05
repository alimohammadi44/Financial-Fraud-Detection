from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .data import Snapshot, TemporalElliptic
from .losses import labeled_focal_loss
from .metrics import binary_fraud_metrics
from .model import FGEGCN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(cfg: Dict) -> torch.device:
    requested = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _snapshot_to_device(snapshot: Snapshot, device: torch.device):
    return (
        snapshot.x.to(device),
        snapshot.y.to(device),
        snapshot.edge_index.to(device),
    )


def _evaluate_sequence(
    model: FGEGCN,
    snapshots: List[Snapshot],
    *,
    eval_start: int,
    eval_end: int,
    device: torch.device,
    keep_artifacts: bool = False,
):
    """Run from the beginning so recurrent weight states see all prior snapshots."""
    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    per_timestep = {}
    artifacts = []

    with torch.no_grad():
        states = [s for s in model.initial_states()]
        for snapshot in snapshots:
            if snapshot.time_step > eval_end:
                break
            x, y, edge_index = _snapshot_to_device(snapshot, device)
            out = model.forward_snapshot(x, edge_index, states)
            states = out["states"]

            if snapshot.time_step < eval_start:
                continue

            labeled = y >= 0
            if labeled.any():
                p = torch.softmax(out["logits"], dim=-1)[:, 1]
                y_np = y[labeled].cpu().numpy()
                p_np = p[labeled].cpu().numpy()
                ys.append(y_np)
                probs.append(p_np)
                per_timestep[str(snapshot.time_step)] = binary_fraud_metrics(y_np, p_np)

            if keep_artifacts:
                artifacts.append(
                    {
                        "time_step": snapshot.time_step,
                        "tx_ids": snapshot.tx_ids,
                        "y": y.cpu(),
                        "logits": out["logits"].cpu(),
                        "z_graph": out["z_graph"].cpu(),
                        "s_feature": out["s_feature"].cpu(),
                        "gamma": out["gamma"].cpu(),
                        "fused": out["fused"].cpu(),
                    }
                )

    if not ys:
        raise RuntimeError(f"No labeled nodes found in evaluation window {eval_start}-{eval_end}")
    y_all = np.concatenate(ys)
    p_all = np.concatenate(probs)
    aggregate = binary_fraud_metrics(y_all, p_all)
    return aggregate, per_timestep, artifacts


def train_reproduction(dataset: TemporalElliptic, cfg: Dict) -> dict:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = _device_from_config(cfg)

    train_max = int(data_cfg.get("train_max_time", 30))
    val_max = int(data_cfg.get("val_max_time", 34))
    test_max = int(data_cfg.get("test_max_time", 49))
    val_start = train_max + 1
    test_start = val_max + 1

    train_snapshots = dataset.between(1, train_max)
    if not train_snapshots:
        raise ValueError("Training window is empty")

    model = FGEGCN(
        input_dim=int(data_cfg.get("local_feature_dim", 94)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 64)),
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    gamma = float(train_cfg.get("focal_gamma", 2.0))
    alpha_licit = float(train_cfg.get("alpha_licit", 0.70))
    alpha_illicit = float(train_cfg.get("alpha_illicit", 0.29))
    epochs = int(train_cfg.get("epochs", 500))
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "illicit_f1"))

    out_dir = Path(cfg.get("output_dir", "outputs/fg_egcn_paper"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        states = [s for s in model.initial_states()]

        loss_sum = None
        labeled_count = 0
        for snapshot in train_snapshots:
            x, y, edge_index = _snapshot_to_device(snapshot, device)
            out = model.forward_snapshot(x, edge_index, states)
            states = out["states"]
            snapshot_loss, n_labeled = labeled_focal_loss(
                out["logits"],
                y,
                gamma=gamma,
                alpha_licit=alpha_licit,
                alpha_illicit=alpha_illicit,
                reduction="sum",
            )
            loss_sum = snapshot_loss if loss_sum is None else loss_sum + snapshot_loss
            labeled_count += n_labeled

        if labeled_count == 0 or loss_sum is None:
            raise RuntimeError("No labeled training nodes were found")
        loss = loss_sum / labeled_count
        loss.backward()
        optimizer.step()

        val_metrics, _, _ = _evaluate_sequence(
            model,
            dataset.snapshots,
            eval_start=val_start,
            eval_end=val_max,
            device=device,
        )
        if checkpoint_metric not in val_metrics:
            raise KeyError(f"Unknown checkpoint metric: {checkpoint_metric}")
        score = float(val_metrics[checkpoint_metric])
        row = {
            "epoch": epoch,
            "train_focal_loss": float(loss.detach().cpu()),
            "validation": val_metrics,
        }
        history.append(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best_model.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} loss={row['train_focal_loss']:.6f} "
                f"val_illicit_f1={val_metrics['illicit_f1']:.4f} "
                f"best={best_score:.4f}@{best_epoch}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)

    test_metrics, test_per_timestep, test_artifacts = _evaluate_sequence(
        model,
        dataset.snapshots,
        eval_start=test_start,
        eval_end=test_max,
        device=device,
        keep_artifacts=True,
    )

    # Export the full temporal evidence streams once using the frozen best model.
    # These tensors are deliberately separated from future quantum code so C1
    # can be frozen before QML training starts.
    _, _, full_artifacts = _evaluate_sequence(
        model,
        dataset.snapshots,
        eval_start=1,
        eval_end=test_max,
        device=device,
        keep_artifacts=True,
    )
    torch.save(full_artifacts, out_dir / "frozen_fg_egcn_evidence.pt")

    result = {
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "checkpoint_metric": checkpoint_metric,
        "split": {
            "train": [1, train_max],
            "validation": [val_start, val_max],
            "test": [test_start, test_max],
        },
        "test": test_metrics,
        "test_per_timestep": test_per_timestep,
    }

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, indent=2)

    return result
