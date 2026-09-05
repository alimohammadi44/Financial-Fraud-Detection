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
- Feature-gated fusion: H = S + Gamma * (Z - S).
- Adam, learning rate 1e-3, max 500 epochs.
- No weight decay unless otherwise stated.
- Focal loss gamma=2.0.
- Alpha values reported as [0.70, 0.29, 0.01] for licit, illicit, unknown.
- Random seed 42.
- Best checkpoint selected by validation illicit F1.
- Published FG-EGCN test metrics: illicit precision 0.834, illicit recall 0.722, illicit F1 0.774, MicroAVG F1 0.971.

## Material details not sufficiently specified

These are explicit reproduction assumptions/sensitivity variables, not claimed author settings.

1. The paper does not state how timesteps 1–34 are subdivided into training and validation despite selecting the best checkpoint by validation illicit F1. Initial choice: train 1–30, validation 31–34.
2. Exact hidden width of the gate MLP is not stated. Initial choice: 64.
3. Exact Top-K summarization implementation inside the EvolveGCN-H weight evolution is not fully specified at software level. We implement the equations directly and record the choice.
4. The public Elliptic feature CSV stores `txId`, `time_step`, and 165 additional numeric columns. The original Elliptic definition counts the time step among the 94 local features. Initial choice: model features are columns 1:95, i.e. time step + the following 93 local attributes; time step is also used to assign temporal snapshots.
5. Exact treatment of inter-snapshot edges is not stated at implementation level. The initial model uses each time-step induced subgraph, matching the temporal-snapshot formulation.
6. Exact focal-loss reduction is not stated. Initial choice: mean over labeled nodes, weighted by the reported licit/illicit alpha entries; unknown nodes do not contribute supervised loss.
7. The paper gives a single fixed seed (42), not a multi-seed uncertainty estimate. For C1, seed 42 is the direct reproduction; for the QIML study, we will add multiple seeds after the baseline is frozen.

## Reproduction discipline

- Do not tune the future quantum module until the FG-EGCN classical implementation and data split are frozen.
- If the published 0.774 illicit F1 is not approached, first test only the documented ambiguity variables above.
- Do not alter the test window to improve results.
- Report illicit precision/recall/F1 exactly, plus PR-AUC and ROC-AUC for the later QIML experiments.
- Keep unknown nodes in message passing but exclude them from supervised loss and labeled evaluation.
