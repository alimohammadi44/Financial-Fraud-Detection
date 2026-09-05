# QIML 2026 — FG-EGCN Reproduction and Quantum Extension Base

This directory is the primary **C1 classical reproduction** for the QIML 2026 study.

Base paper:

> Na Han, Ruixian Zhang, Xiaoyun Liu, Haining Zhang, **"Illicit Bitcoin transaction detection via feature-gated temporal graph learning,"** *Scientific Reports* 16, 22352 (2026). DOI: `10.1038/s41598-026-53783-y`.

The paper is used as a strong recent classical baseline because it gives explicit equations for its feature gate, temporal EvolveGCN-H backbone, focal-loss settings, chronological evaluation, and headline illicit-class metrics.

This is an **independent reimplementation from the published method**, not author code.

## Published target on Elliptic

The paper reports for FG-EGCN:

- illicit precision: **0.834**
- illicit recall: **0.722**
- illicit F1: **0.774**
- micro-average F1: **0.971**

These are reproduction targets, not results claimed by this repository until the original Elliptic data are run.

## Elliptic data

Use the original Elliptic distribution and place:

```text
data/raw/elliptic/
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv
```

The dataset is intentionally not committed.

## Reproduction protocol

The implementation follows the paper where specified:

- all nodes, including unknown-label nodes, remain in the graph as structural context;
- supervised loss is computed only on licit/illicit labeled nodes;
- first **94 local features** are used;
- 49 temporal snapshots are preserved;
- timesteps 35–49 are held out as the chronological test period;
- two temporal graph-convolution layers;
- hidden dimension 64;
- dropout 0.5;
- residual connections and LayerNorm;
- feature branch 94 -> 64;
- scalar learned gate blends feature and temporal-graph branches;
- Adam, learning rate 1e-3, max 500 epochs;
- focal loss with gamma=2 and the paper's class alpha values;
- seed 42;
- checkpoint selection by validation illicit F1.

The article states that timesteps 1–34 are used for training/model selection but does not specify the validation subdivision. The initial reproduction uses **1–30 train, 31–34 validation, 35–49 test** as an explicit assumption. See `REPRODUCIBILITY_NOTES.md`.

## Run

```bash
export PYTHONPATH=$PWD/qiml2026_fg_egcn/src
python -m fg_egcn.cli --config qiml2026_fg_egcn/configs/paper.yaml
```

## Why this base is useful for the quantum contribution

FG-EGCN exposes two evidence streams for every transaction:

- `S`: direct/local feature evidence;
- `Z`: temporal graph evidence.

Its published gate forms:

```text
H = S + Gamma * (Z - S)
```

so `||Z-S||` directly measures feature-vs-graph disagreement. After C1 is frozen, the QIML extension will test whether a small VQC should be invoked **only for nodes with high evidence disagreement and/or high predictive uncertainty**, under a fixed quantum-call budget.

The planned quantum stage therefore does not replace FG-EGCN. It adds a controlled residual refinement head and compares it against a matched classical refiner, random routing, and full quantum routing.
