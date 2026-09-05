# QIML 2026 — Independent ChronoWave-GNN Reproduction

This directory is the **classical C1 reproduction stage** for a proposed QIML 2026 study on resource-aware quantum refinement of temporal graph fraud models.

The base paper is:

> Ziqian Lin, Qining Luo, Dongze Wu, et al. "Detecting illicit transactions in bitcoin: a wavelet-temporal graph transformer approach for anti-money laundering." *Scientific Reports* 16, 1548 (2026). DOI: `10.1038/s41598-025-23901-3`.

This is an **independent reimplementation from the published method**, not the authors' code.

## Why reproduce first?

The eventual quantum experiment must be compared against a strong, frozen classical model. We therefore first reproduce the paper's documented Elliptic pipeline and identify any under-specified implementation details. Only after the baseline is validated will the quantum uncertainty-gated residual module be added.

## Data

Download the original Elliptic dataset from the official Kaggle distribution and place these licensed files under:

```text
data/raw/elliptic/
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv
```

The dataset is intentionally not committed to this repository.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r qiml2026_chronowave/requirements.txt
```

For GPU use, install the PyTorch/PyG wheels matching your CUDA environment.

Set the source directory:

```bash
export PYTHONPATH=$PWD/qiml2026_chronowave/src
```

## Run the paper-style reproduction

```bash
python -m chronowave.cli \
  --config qiml2026_chronowave/configs/paper_stratified.yaml
```

The paper reports Elliptic test accuracy **0.9802** and F1 **0.9799**. We will compare our five-seed results to these values but will not silently tune under-specified hyperparameters to force agreement.

## Run the chronological robustness protocol

```bash
python -m chronowave.cli \
  --config qiml2026_chronowave/configs/chronological.yaml
```

This is not the primary reproduction number. It is the leakage-resistant temporal protocol intended for the subsequent QIML contribution.

## Current model pipeline

```text
Elliptic labeled transactions
  -> train-fitted standardization of raw features
  -> level-2 Haar approximation coefficients
  -> train-fitted standardization of wavelet coefficients
  -> concatenate raw + wavelet features
  -> 8-D sinusoidal time encoding + learnable projection
  -> 3 x PyG TransformerConv + ELU + dropout
  -> linear 2-class classifier
```

## Important reproducibility gaps

The article does not fully specify several implementation details, notably the Transformer hidden width and attention-head count. The initial configuration uses **128 hidden channels and 4 heads as explicit assumptions**, not as claimed settings from Lin et al. See `REPRODUCIBILITY_NOTES.md` before interpreting results.

## Next stage after C1 is frozen

The QIML contribution will add a small VQC only to transactions selected by calibrated predictive uncertainty under a fixed quantum-call budget, with a parameter/dimension-matched classical refiner as the main control. That code is deliberately **not** mixed into the reproduction baseline yet.
