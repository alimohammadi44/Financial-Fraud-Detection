# FG-EGCN reproducibility notes

Base paper: Han et al., *Scientific Reports* 16, 22352 (2026), DOI 10.1038/s41598-026-53783-y.

## Directly specified by the paper

- Elliptic transaction graph: 203,769 nodes, 234,355 directed edges, 49 time steps.
- Unknown-label nodes are retained for graph context; supervised loss is restricted to labeled nodes.
- First 94 local features are used.
- Chronological protocol: first 34 time steps for training/model selection; final 15 time steps for testing.
- EvolveGCN-H-style temporal graph encoder.
- Two temporal graph-convolution layers.
- Hidden dimension 64.
- Feature projector: 94 -> 64.
- Dropout 0.5.
- Residual connections and LayerNorm.
- Self-loops followed by symmetric adjacency normalization.
- Feature-gated fusion: H = S + Gamma * (Z - S).
- Adam, learning rate 1e-3, max 500 epochs.
- No weight decay unless otherwise stated.
- Focal loss gamma=2.0.
- Alpha values reported as [0.70, 0.29, 0.01] for licit, illicit, unknown.
- Random seed 42.
- Best checkpoint selected by validation illicit F1.
- Published FG-EGCN test metrics: illicit precision 0.834, illicit recall 0.722, illicit F1 0.774, MicroAVG F1 0.971.

## EvolveGCN-H fidelity

Because FG-EGCN explicitly adopts EvolveGCN-H, the implementation was cross-checked against Pareja et al. (AAAI 2020) and IBM's archived official EvolveGCN implementation. We therefore use a **matrix-GRU**, not an ordinary vector/row-wise GRU. The Top-K scorer is normalized and selected node rows are weighted by `tanh(score)`, consistent with EvolveGCN-H. Han et al. specify `k=d_l` and allow a linear projection when the summary shape does not match `W_t`; we follow that FG-EGCN-specific rule before the matrix-GRU update.

## Material details not sufficiently specified

These are explicit reproduction assumptions/sensitivity variables, not claimed author settings.

1. The paper does not state how timesteps 1–34 are subdivided into training and validation despite selecting the best checkpoint by validation illicit F1. Initial choice: **train 1–29, validation 30–34**. This preserves the authors' 35–49 test window and provides a five-timestep validation block. The alternative 1–30 / 31–34 split is retained as a sensitivity check if necessary.
2. Exact hidden width of the gate MLP is not stated. Initial choice: 64.
3. The paper denotes the feature/gate nonlinearity by `delta` without naming the function. Initial choice: ReLU.
4. The public Elliptic feature CSV stores `txId`, `time_step`, and 165 additional numeric columns. The original Elliptic definition counts the time step among the 94 local features. Initial choice: model features are columns 1:95, i.e. time step + the following 93 local attributes; time step is also used to assign temporal snapshots.
5. Exact sparse orientation used to implement symmetric normalization on the directed adjacency is not published. We preserve listed source->target edges, add self-loops, and use target-degree source-to-target normalization; graph-orientation sensitivity must be checked if the target result is not approached.
6. Exact focal-loss reduction is not stated. Initial choice: mean over labeled nodes across the complete temporal training window, weighted by the reported licit/illicit alpha entries; unknown nodes do not contribute supervised loss.
7. The paper gives a single fixed seed (42), not a multi-seed uncertainty estimate. For C1, seed 42 is the direct reproduction; for the QIML study, we will add multiple seeds after the baseline is frozen.
8. The textual architecture shorthand can be read as `94 -> 64 -> C`, while the fusion equations require the temporal representation `Z` and feature representation `S` to share hidden dimension `h` before the classifier. The implementation therefore uses two temporal graph layers with hidden dimension 64 (`94 -> 64 -> 64`) followed by the 64-to-2 classifier; this is the equation-consistent interpretation.

## Reproduction discipline

- Do not tune the future quantum module until the FG-EGCN classical implementation and data split are frozen.
- If the published 0.774 illicit F1 is not approached, first test only the documented ambiguity variables above.
- Do not alter the test window to improve results.
- Report illicit precision/recall/F1 exactly, plus PR-AUC and ROC-AUC for the later QIML experiments.
- Keep unknown nodes in message passing but exclude them from supervised loss and labeled evaluation.
