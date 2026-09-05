from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .features import haar_approximation_level2


@dataclass
class GraphBundle:
    x: torch.Tensor
    time_step: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_index: torch.Tensor
    train_edge_index: torch.Tensor
    val_edge_index: torch.Tensor
    test_edge_index: torch.Tensor
    tx_ids: np.ndarray
    metadata: Dict[str, object]


def _standardize_fit_train(values: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    train = values[train_mask]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return ((values - mean) / std).astype(np.float32), {
        "mean": mean.squeeze(0).astype(np.float32),
        "std": std.squeeze(0).astype(np.float32),
    }


def _paper_stratified_split(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, random_state=seed, stratify=y[rest_idx])
    masks = []
    for chosen in (train_idx, val_idx, test_idx):
        mask = np.zeros(len(y), dtype=bool)
        mask[chosen] = True
        masks.append(mask)
    return tuple(masks)


def _chronological_split(time_step: np.ndarray, train_max: int, val_max: int):
    train = time_step <= train_max
    val = (time_step > train_max) & (time_step <= val_max)
    test = time_step > val_max
    if not train.any() or not val.any() or not test.any():
        raise ValueError(f"Chronological split empty: train={train.sum()}, val={val.sum()}, test={test.sum()}")
    return train, val, test


def _edge_subset(edge_index: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    keep = allowed[edge_index[0]] & allowed[edge_index[1]]
    return edge_index[:, keep]


def load_elliptic(
    data_dir: str | Path,
    *,
    split: str = "paper_stratified",
    seed: int = 42,
    train_max_time: int = 30,
    val_max_time: int = 34,
    include_timestep_in_raw_features: bool = False,
) -> GraphBundle:
    """Load original Elliptic following the paper's documented core protocol.

    Unknown labels are removed before graph construction. Directed edges are
    retained only if both endpoints remain. No preprocessing self-loops are
    introduced. Raw and level-2 Haar approximation features are standardized
    using training-node statistics and concatenated.
    """
    data_dir = Path(data_dir)
    feature_path = data_dir / "elliptic_txs_features.csv"
    class_path = data_dir / "elliptic_txs_classes.csv"
    edge_path = data_dir / "elliptic_txs_edgelist.csv"
    for path in (feature_path, class_path, edge_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Elliptic file: {path}")

    feats = pd.read_csv(feature_path, header=None)
    classes = pd.read_csv(class_path)
    edges = pd.read_csv(edge_path)

    known = classes[classes["class"].astype(str) != "unknown"].copy()
    known["y"] = known["class"].astype(str).map({"1": 1, "2": 0})
    if known["y"].isna().any():
        raise ValueError(f"Unexpected class values: {known.loc[known['y'].isna(), 'class'].unique().tolist()}")

    tx_to_feat_row = pd.Series(np.arange(len(feats)), index=feats.iloc[:, 0].astype(np.int64)).to_dict()
    known_tx = known["txId"].astype(np.int64).to_numpy()
    missing = [int(tx) for tx in known_tx if int(tx) not in tx_to_feat_row]
    if missing:
        raise ValueError(f"{len(missing)} labeled txIds are absent from feature file")

    rows = np.asarray([tx_to_feat_row[int(tx)] for tx in known_tx], dtype=np.int64)
    selected = feats.iloc[rows].reset_index(drop=True)
    time_step = selected.iloc[:, 1].to_numpy(dtype=np.int64)
    raw_start = 1 if include_timestep_in_raw_features else 2
    raw = selected.iloc[:, raw_start:].to_numpy(dtype=np.float32)
    y = known["y"].to_numpy(dtype=np.int64)

    if split == "paper_stratified":
        train_mask, val_mask, test_mask = _paper_stratified_split(y, seed)
    elif split == "chronological":
        train_mask, val_mask, test_mask = _chronological_split(time_step, train_max_time, val_max_time)
    else:
        raise ValueError("split must be 'paper_stratified' or 'chronological'")

    wave = haar_approximation_level2(raw)
    raw_std, raw_scaler = _standardize_fit_train(raw, train_mask)
    wave_std, wave_scaler = _standardize_fit_train(wave, train_mask)
    x = np.concatenate([raw_std, wave_std], axis=1).astype(np.float32)

    node_index = {int(tx): i for i, tx in enumerate(known_tx)}
    src = edges.iloc[:, 0].astype(np.int64).to_numpy()
    dst = edges.iloc[:, 1].astype(np.int64).to_numpy()
    pairs = [(node_index[int(a)], node_index[int(b)]) for a, b in zip(src, dst)
             if int(a) in node_index and int(b) in node_index and int(a) != int(b)]
    if not pairs:
        raise ValueError("No edges remain after filtering to known labeled transactions")
    edge_index = np.unique(np.asarray(pairs, dtype=np.int64).T, axis=1)

    if split == "chronological":
        train_edge = _edge_subset(edge_index, train_mask)
        val_edge = _edge_subset(edge_index, train_mask | val_mask)
        test_edge = _edge_subset(edge_index, train_mask | val_mask | test_mask)
    else:
        # Matches the article's transductive random-node split: same labeled graph,
        # while loss and metrics are restricted by node masks.
        train_edge = val_edge = test_edge = edge_index

    metadata = {
        "split": split,
        "seed": seed,
        "n_nodes": int(len(y)),
        "n_illicit": int((y == 1).sum()),
        "n_licit": int((y == 0).sum()),
        "n_edges": int(edge_index.shape[1]),
        "raw_feature_dim": int(raw.shape[1]),
        "wavelet_dim": int(wave.shape[1]),
        "model_input_without_time_dim": int(x.shape[1]),
        "include_timestep_in_raw_features": include_timestep_in_raw_features,
        "standardization": "train-only",
        "raw_scaler": raw_scaler,
        "wave_scaler": wave_scaler,
    }

    return GraphBundle(
        x=torch.from_numpy(x), time_step=torch.from_numpy(time_step), y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask), val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask), edge_index=torch.from_numpy(edge_index).long(),
        train_edge_index=torch.from_numpy(train_edge).long(), val_edge_index=torch.from_numpy(val_edge).long(),
        test_edge_index=torch.from_numpy(test_edge).long(), tx_ids=known_tx, metadata=metadata,
    )
