from __future__ import annotations
import math
import numpy as np

def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = midranks
    return result

def fast_delong(
    predictions_sorted: np.ndarray, positive_count: int
) -> tuple[np.ndarray, np.ndarray]:
    model_count, total_count = predictions_sorted.shape
    negative_count = total_count - positive_count
    positive = predictions_sorted[:, :positive_count]
    negative = predictions_sorted[:, positive_count:]

    positive_midrank = np.empty((model_count, positive_count), dtype=float)
    negative_midrank = np.empty((model_count, negative_count), dtype=float)
    total_midrank = np.empty((model_count, total_count), dtype=float)
    for model_index in range(model_count):
        positive_midrank[model_index] = compute_midrank(positive[model_index])
        negative_midrank[model_index] = compute_midrank(negative[model_index])
        total_midrank[model_index] = compute_midrank(
            predictions_sorted[model_index]
        )

    aucs = (
        total_midrank[:, :positive_count].sum(axis=1)
        / positive_count
        / negative_count
        - (positive_count + 1.0) / 2.0 / negative_count
    )
    v01 = (
        total_midrank[:, :positive_count] - positive_midrank
    ) / negative_count
    v10 = 1.0 - (
        total_midrank[:, positive_count:] - negative_midrank
    ) / positive_count
    covariance = (
        np.atleast_2d(np.cov(v01, bias=False)) / positive_count
        + np.atleast_2d(np.cov(v10, bias=False)) / negative_count
    )
    return aucs, covariance

def paired_delong(
    y_true: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[float, float, float, float, float]:
    y_true = np.asarray(y_true)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if y_true.ndim != 1 or first.shape != y_true.shape or second.shape != y_true.shape:
        raise ValueError('Paired DeLong requires aligned one-dimensional labels and scores')
    if not np.isin(y_true, [0, 1]).all() or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Paired DeLong requires observed binary labels and finite scores')
    if min(np.sum(y_true == 0), np.sum(y_true == 1)) < 2:
        raise ValueError('Paired DeLong covariance requires at least two patients in each class')
    order = np.argsort(-y_true)
    predictions = np.vstack([first, second])[:, order]
    aucs, covariance = fast_delong(predictions, int(y_true.sum()))
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        return float(aucs[0]), float(aucs[1]), difference, np.nan, np.nan
    z_value = difference / np.sqrt(variance)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return (
        float(aucs[0]),
        float(aucs[1]),
        difference,
        float(z_value),
        float(p_value),
    )
