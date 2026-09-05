from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


UNKNOWN_LABEL = -1
LICIT_LABEL = 0
ILLICIT_LABEL = 1


@dataclass
class Snapshot:
    time_step: int
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    tx_ids: np.ndarray

    @property
    def labeled_mask(self) -> torch.Tensor:
        return self.y >= 0


@dataclass
class TemporalElliptic:
    snapshots: List[Snapshot]
    metadata: Dict[str, object]

    def between(self, start: int, end: int) -> List[Snapshot]:
        return [s for s in self.snapshots if start <= s.time_step <= end]


def _map_labels(classes: pd.DataFrame) -> Dict[int, int]:
    if "txId" not in classes.columns or "class" not in classes.columns:
        raise ValueError("Expected elliptic_txs_classes.csv columns: txId,class")

    mapping: Dict[int, int] = {}
    for tx, raw in zip(classes["txId"], classes["class"]):
        value = str(raw).strip().lower()
        if value == "1":
            y = ILLICIT_LABEL
        elif value == "2":
            y = LICIT_LABEL
        elif value == "unknown":
            y = UNKNOWN_LABEL
        else:
            raise ValueError(f"Unexpected Elliptic class value: {raw!r}")
        mapping[int(tx)] = y
    return mapping


def load_elliptic_temporal(
    data_dir: str | Path,
    *,
    local_feature_dim: int = 94,
) -> TemporalElliptic:
    """Load Elliptic as 49 induced temporal transaction snapshots.

    FG-EGCN keeps unknown-label nodes in the graph and restricts supervised loss
    to licit/illicit nodes. The public feature CSV contains `txId`, `time_step`,
    then 165 additional numeric values. The original Elliptic definition counts
    time among the 94 local descriptors. Accordingly, the initial reproduction
    uses CSV columns 1:95 as the 94-dimensional local feature vector while also
    using column 1 to assign each transaction to its temporal snapshot.

    No additional feature scaling is applied because the FG-EGCN article does
    not report one; this avoids silently introducing an extra preprocessing step.
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

    if feats.shape[1] < 1 + local_feature_dim:
        raise ValueError(
            f"Feature file has {feats.shape[1]} columns; cannot select "
            f"txId plus {local_feature_dim} local features"
        )

    tx_ids = feats.iloc[:, 0].to_numpy(dtype=np.int64)
    time_steps = feats.iloc[:, 1].to_numpy(dtype=np.int64)
    # Columns 1..94 inclusive => 94 local features, including time_step.
    x_all = feats.iloc[:, 1 : 1 + local_feature_dim].to_numpy(dtype=np.float32)

    label_map = _map_labels(classes)
    y_all = np.asarray([label_map.get(int(tx), UNKNOWN_LABEL) for tx in tx_ids], dtype=np.int64)

    # Build a lookup for edge assignment and local indexing inside each snapshot.
    tx_to_time: Dict[int, int] = {int(tx): int(t) for tx, t in zip(tx_ids, time_steps)}
    indices_by_time: Dict[int, np.ndarray] = {
        int(t): np.flatnonzero(time_steps == t) for t in sorted(np.unique(time_steps))
    }
    local_index: Dict[int, int] = {}
    for t, global_rows in indices_by_time.items():
        for local, global_row in enumerate(global_rows):
            local_index[int(tx_ids[global_row])] = local

    src_col = "txId1" if "txId1" in edges.columns else edges.columns[0]
    dst_col = "txId2" if "txId2" in edges.columns else edges.columns[1]

    edge_pairs_by_time: Dict[int, list[tuple[int, int]]] = {
        int(t): [] for t in indices_by_time
    }
    cross_time_edges = 0
    unknown_endpoint_edges = 0
    for src_raw, dst_raw in zip(edges[src_col], edges[dst_col]):
        src = int(src_raw)
        dst = int(dst_raw)
        t_src = tx_to_time.get(src)
        t_dst = tx_to_time.get(dst)
        if t_src is None or t_dst is None:
            unknown_endpoint_edges += 1
            continue
        if t_src != t_dst:
            # FG-EGCN is formulated as per-time-step graph snapshots. Cross-time
            # edges cannot be placed inside an induced A_t and are recorded.
            cross_time_edges += 1
            continue
        if src == dst:
            # Model adds self loops during normalized propagation.
            continue
        edge_pairs_by_time[t_src].append((local_index[src], local_index[dst]))

    snapshots: List[Snapshot] = []
    total_snapshot_edges = 0
    for t in sorted(indices_by_time):
        rows = indices_by_time[t]
        pairs = edge_pairs_by_time[t]
        if pairs:
            edge_np = np.asarray(pairs, dtype=np.int64).T
            # Remove exact duplicate directed edges while preserving direction.
            edge_np = np.unique(edge_np, axis=1)
        else:
            edge_np = np.empty((2, 0), dtype=np.int64)
        total_snapshot_edges += edge_np.shape[1]
        snapshots.append(
            Snapshot(
                time_step=int(t),
                x=torch.from_numpy(x_all[rows]),
                y=torch.from_numpy(y_all[rows]),
                edge_index=torch.from_numpy(edge_np).long(),
                tx_ids=tx_ids[rows],
            )
        )

    metadata = {
        "n_nodes": int(len(tx_ids)),
        "n_raw_edges": int(len(edges)),
        "n_snapshot_edges": int(total_snapshot_edges),
        "cross_time_edges_dropped": int(cross_time_edges),
        "unknown_endpoint_edges_dropped": int(unknown_endpoint_edges),
        "n_timesteps": int(len(snapshots)),
        "time_min": int(min(indices_by_time)),
        "time_max": int(max(indices_by_time)),
        "feature_dim": int(local_feature_dim),
        "n_licit": int((y_all == LICIT_LABEL).sum()),
        "n_illicit": int((y_all == ILLICIT_LABEL).sum()),
        "n_unknown": int((y_all == UNKNOWN_LABEL).sum()),
        "feature_preprocessing": "none beyond selecting first 94 local features",
    }
    return TemporalElliptic(snapshots=snapshots, metadata=metadata)
