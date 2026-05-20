"""Classification metric helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def classification_summary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Return accuracy and macro-F1 without requiring scikit-learn."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction list.")

    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    f1_values: list[float] = []
    for label in labels:
        tp = sum(int(a == label and b == label) for a, b in zip(y_true, y_pred))
        fp = sum(int(a != label and b == label) for a, b in zip(y_true, y_pred))
        fn = sum(int(a == label and b != label) for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)

    per_label_count = defaultdict(int)
    for label in y_true:
        per_label_count[int(label)] += 1

    return {
        "accuracy": correct / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "num_samples": float(len(y_true)),
        "num_classes": float(len(labels)),
    }
