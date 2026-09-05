from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_fraud_metrics(y_true, prob_illicit, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_illicit = np.asarray(prob_illicit, dtype=float)
    if len(y_true) == 0:
        raise ValueError("No labeled observations supplied to metrics")
    pred = (prob_illicit >= threshold).astype(np.int64)

    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "illicit_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "illicit_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "illicit_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob_illicit))
        out["pr_auc"] = float(average_precision_score(y_true, prob_illicit))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out
