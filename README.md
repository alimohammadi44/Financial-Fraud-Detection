# Financial Fraud Detection

Machine learning methods and experiments for detecting financial fraud using structured transaction data and fraud-risk modeling workflows.

## Repository Category

**Original / Portfolio Project**

This repository is part of my public machine learning portfolio. It focuses on practical fraud detection workflows, model comparison, and future extensions toward graph-based and time-aware fraud detection.

## Overview

Financial fraud detection is a highly imbalanced classification problem where fraudulent activity is rare but costly. This repository explores machine learning approaches for identifying suspicious transactions and improving fraud-risk prediction.

The project can be used as a starting point for:

- binary fraud classification
- model comparison on imbalanced data
- feature engineering for transaction datasets
- evaluation with fraud-appropriate metrics
- future graph-based fraud detection experiments

## Main Topics

- Financial fraud detection
- Imbalanced classification
- Machine learning for tabular data
- Model evaluation with AUC, precision, recall, and F1-score
- Transaction-risk modeling
- Future graph-based fraud detection extensions

## Possible Methods

Depending on the available notebooks and experiments, this project may include or be extended with:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Neural networks
- Anomaly detection methods
- Graph Neural Networks for relational fraud patterns

## Suggested Project Structure

```text
.
├── data/             # Dataset files or dataset download instructions
├── notebooks/        # Jupyter notebooks for experiments
├── src/              # Reusable Python source code
├── models/           # Saved models, if applicable
├── reports/          # Results, plots, and evaluation reports
├── requirements.txt  # Python dependencies
└── README.md
```

## Evaluation Metrics

Fraud datasets are usually highly imbalanced, so accuracy alone is not enough. Useful metrics include:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Cost-sensitive evaluation

## How to Use

Clone the repository:

```bash
git clone https://github.com/alimohammadi44/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

Install dependencies when a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Then run the available notebooks or Python scripts.

## Future Work

- Add stronger baseline models
- Add graph-based fraud detection experiments
- Add time-series or sequence-based fraud features
- Improve model explainability with SHAP or feature importance
- Add reproducible experiment tracking

## Author

Ali Mohammadi — [@alimohammadi44](https://github.com/alimohammadi44)
