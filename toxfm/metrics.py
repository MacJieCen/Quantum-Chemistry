"""The manuscript's eight metrics, with explicit single-class handling."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             roc_auc_score, roc_curve)

METRIC_NAMES = ("Accuracy", "Specificity", "Sensitivity", "Precision",
                "AUCROC", "AUPR", "F1", "BalanceACC")


def evaluate(y, scores, threshold):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must be finite, non-empty and aligned")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Only binary labels 0/1 are supported")
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    two_classes = len(np.unique(y)) == 2
    return {"Accuracy": float(accuracy_score(y, predictions)),
            "Specificity": specificity, "Sensitivity": sensitivity,
            "Precision": float(precision_score(y, predictions, zero_division=0)),
            "AUCROC": float(roc_auc_score(y, scores)) if two_classes else None,
            "AUPR": float(average_precision_score(y, scores)) if two_classes else None,
            "F1": float(f1_score(y, predictions, zero_division=0)),
            "BalanceACC": (specificity + sensitivity) / 2 if two_classes else None,
            "threshold": float(threshold), "n": len(y),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_threshold(y, scores):
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection needs both validation classes")
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if not valid.any():
        raise ValueError("No valid validation threshold")
    bacc = (tpr[valid] + 1 - fpr[valid]) / 2
    candidates = thresholds[valid][np.isclose(bacc, bacc.max())]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])
