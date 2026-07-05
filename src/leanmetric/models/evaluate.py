from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    - y_true : array-like, required
        Ground truth target values.
    - y_pred : array-like, required
        Predicted target values.

    Returns
    -------
    - metrics : Dict[str, float]
        Dictionary containing regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }