from __future__ import annotations

import joblib
import pandas as pd


def predict_from_dataframe(
    df: pd.DataFrame,
    model_path: str = "models/best_model.joblib",
):
    """
    Generate predictions using a trained model pipeline.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe containing the same training features.
    - model_path : str, required
        Path to the serialized model.

    Returns
    -------
    - predictions : np.ndarray
        Predicted body fat values.
    """
    model = joblib.load(model_path)
    predictions = model.predict(df)
    return predictions