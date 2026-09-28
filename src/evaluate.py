import numpy as np


def calculate_mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute deviation, equivalent to mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))