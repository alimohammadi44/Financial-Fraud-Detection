# ChronoWave-GNN reproduction notes

Base paper:

Z. Lin et al., **"Detecting illicit transactions in bitcoin: a wavelet-temporal graph transformer approach for anti-money laundering,"** *Scientific Reports* 16, 1548 (2026). DOI: 10.1038/s41598-025-23901-3.

## Directly specified by the paper

- Elliptic transaction graph.
- Unknown labels filtered before the main supervised graph is constructed.
- Directed edges retained only when both endpoint transactions are in the retained labeled set.
- No preprocessing self-loops.
- Main reproduction split: stratified 80/10/10.
- Level-2 Haar DWT; approximation coefficients retained.
- Raw and wavelet feature blocks standardized before concatenation.
- 8-dimensional sinusoidal timestamp encoding followed by a learnable linear projection.
- TGAT+ described as a 3-layer `TransformerConv` backbone.
- ELU activation and dropout 0.4.
- AdamW, learning rate 0.005, weight decay 5e-4.
- Maximum 200 epochs, cosine learning-rate schedule.
- Label smoothing 0.1.
- Early stopping on validation F1, patience 20.
- Five random seeds.
- Reported Elliptic test accuracy 0.9802 and F1 0.9799.

## Material details not sufficiently specified in the accessible article

These are *not* presented here as author settings. They are explicit reproduction assumptions and must be sensitivity-tested before claiming reproduction.

1. Transformer hidden width.
2. Number of attention heads.
3. Exact PyG `TransformerConv` options (e.g. root skip/beta/concat details).
4. DWT boundary-extension mode for the odd-length raw feature vector.
5. Whether the Elliptic `time_step` column is counted among the stated 166 raw node features and simultaneously used as timestamp encoding.
6. Minimum learning rate (`eta_min`) in cosine annealing.
7. Exact definition/averaging convention for reported Precision/Recall/F1.
8. Exact random seeds.
9. The article's core equations/pseudocode specify concatenation of raw+wavelet+time features, while later analysis discusses dynamic wavelet fusion/gate activations without a complete reproducible gate formula in the core method. This implementation therefore reproduces the documented concatenation core first and does not invent an undocumented gate.

## Conservative choices in this implementation

- Standardization mean/std are fitted on training nodes only and then applied to validation/test nodes.
- We report illicit-class F1, macro F1, weighted F1, ROC-AUC and PR-AUC so the paper's unspecified F1 convention can be diagnosed rather than guessed.
- The chronological configuration restricts graph edges by information phase (train only; train+validation; all observed test nodes) to reduce temporal leakage.

## Reproduction decision rule

We will not tune the quantum extension until the classical reproduction is frozen. Any deviation from the reported ChronoWave result must be accompanied by a documented sensitivity analysis over the unspecified implementation choices above.
