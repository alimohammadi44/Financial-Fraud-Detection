# FG-EGCN C1 reproduction results

Date: 2026-09-04/05 (America/Toronto / UTC)

Base paper: Na Han, Ruixian Zhang, Xiaoyun Liu, Haining Zhang, "Illicit Bitcoin transaction detection via feature-gated temporal graph learning," Scientific Reports 16, 22352 (2026), DOI 10.1038/s41598-026-53783-y.

These are real executions on the original Elliptic dataset (203,769 nodes, 234,355 edges, 49 timesteps), seed 42, 500 epochs, chronological test window 35-49. The paper does not disclose the internal validation split within timesteps 1-34; these runs use train 1-29, validation 30-34, test 35-49.

## Published target vs independent reproduction

| Metric | Paper FG-EGCN | Run A: literal reported alpha (licit=0.70, illicit=0.29) | Run B: controlled alpha-swap sensitivity (licit=0.29, illicit=0.70) |
|---|---:|---:|---:|
| Illicit precision | 0.834 | 0.29944 | 0.30714 |
| Illicit recall | 0.722 | 0.24746 | 0.55217 |
| Illicit F1 | 0.774 | 0.27098 | 0.39472 |
| Micro-F1 | 0.971 | 0.91350 | 0.88998 |
| ROC-AUC | not headline-reported | 0.85171 | 0.84153 |
| PR-AUC | not headline-reported | 0.27121 | 0.20502 |

## Run A — literal paper wording

GitHub Actions run: 33939192499  
Artifact: 9961364184 (`fg-egcn-reproduction-results`)  
Commit: `9d57208f52be2c799d7a13455398a4faed019f68`

- best epoch: 263
- best validation illicit F1: 0.89698
- test illicit precision: 0.29944
- test illicit recall: 0.24746
- test illicit F1: 0.27098
- test micro-F1: 0.91350
- test ROC-AUC: 0.85171
- test PR-AUC: 0.27121

The training step completed successfully. The workflow status was marked failed only because the comparison script was called without its required `--result` flag; the model run and result artifact were successfully produced.

## Run B — class-weight-order sensitivity

GitHub Actions run: 33939732711  
Artifact: 9961517269 (`fg-egcn-alpha-swapped-results`)  
Commit: `c0dd828e9704ce80be529ea0fccaa8d8a70c6bd6`

Only the first two focal-loss weights were swapped; all other settings remained unchanged.

- best epoch: 356
- best validation illicit F1: 0.89057
- test illicit precision: 0.30714
- test illicit recall: 0.55217
- test illicit F1: 0.39472
- test micro-F1: 0.88998
- test ROC-AUC: 0.84153
- test PR-AUC: 0.20502

## Interpretation

Neither run reproduces the paper's reported illicit F1 of 0.774. Swapping the focal-loss weights raises test illicit F1 from 0.271 to 0.395 mainly by increasing recall, but leaves a large gap. Therefore the current C1 implementation must not yet be described as a successful reproduction and the quantum module should not be used to claim improvement over FG-EGCN until the classical discrepancy is better resolved.

The next controlled checks should target only under-specified implementation choices: validation subdivision within timesteps 1-34, exact adjacency/orientation normalization, the unspecified gate activation/MLP details, and exact EvolveGCN-H implementation details/version behavior. The held-out test window 35-49 must remain fixed.
