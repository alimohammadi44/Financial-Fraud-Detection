from pathlib import Path

import numpy as np
import pandas as pd

from fg_egcn.data import load_elliptic_temporal


def test_temporal_loader_retains_unknown_and_drops_cross_time_edges(tmp_path: Path):
    # Public Elliptic shape: txId + 166 numeric features; column 1 is time_step.
    rows = []
    for tx, t in [(10, 1), (11, 1), (12, 1), (20, 2), (21, 2), (22, 2)]:
        attrs = [float(t)] + [float(tx % 7)] * 165
        rows.append([tx] + attrs)
    pd.DataFrame(rows).to_csv(tmp_path / "elliptic_txs_features.csv", header=False, index=False)

    pd.DataFrame(
        {
            "txId": [10, 11, 12, 20, 21, 22],
            "class": ["1", "2", "unknown", "2", "1", "unknown"],
        }
    ).to_csv(tmp_path / "elliptic_txs_classes.csv", index=False)

    # Four within-time edges and one cross-time edge (12 -> 20).
    pd.DataFrame(
        {"txId1": [10, 11, 20, 21, 12], "txId2": [11, 12, 21, 22, 20]}
    ).to_csv(tmp_path / "elliptic_txs_edgelist.csv", index=False)

    ds = load_elliptic_temporal(tmp_path, local_feature_dim=94)
    assert len(ds.snapshots) == 2
    assert ds.metadata["n_nodes"] == 6
    assert ds.metadata["n_illicit"] == 2
    assert ds.metadata["n_licit"] == 2
    assert ds.metadata["n_unknown"] == 2
    assert ds.metadata["cross_time_edges_dropped"] == 1
    assert ds.metadata["n_snapshot_edges"] == 4
    assert ds.snapshots[0].x.shape == (3, 94)
    assert ds.snapshots[0].labeled_mask.sum().item() == 2
