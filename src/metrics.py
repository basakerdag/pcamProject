import numpy as np
from sklearn.metrics import (balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def pick_threshold(y_true, y_prob, metric="balanced_accuracy"):
    """Pick the decision threshold on the validation set only."""
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        s = balanced_accuracy_score(y_true, pred) if metric == "balanced_accuracy" else f1_score(y_true, pred)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t

def compute_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }