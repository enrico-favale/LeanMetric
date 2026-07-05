from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


TARGET_COL = "bodyfat"


def load_model(model_path: str):
    """
    Load a trained model pipeline.

    Parameters
    ----------
    - model_path : str, required
        Path to the serialized model file.

    Returns
    -------
    - model : object
        Loaded trained model.
    """
    return joblib.load(model_path)


def get_feature_names_from_pipeline(pipeline, X: pd.DataFrame) -> list[str]:
    """
    Extract transformed feature names from a fitted preprocessing pipeline.

    Parameters
    ----------
    - pipeline : object, required
        Fitted sklearn pipeline.
    - X : pd.DataFrame, required
        Original input dataframe.

    Returns
    -------
    - feature_names : list[str]
        Transformed feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = list(X.columns)

    if not feature_names:
        feature_names = list(X.columns)

    return feature_names


def compute_permutation_importance(
    df: pd.DataFrame,
    model_path: str = "models/best_tuned_model.joblib",
    output_dir: str = "models",
    n_repeats: int = 30,
    random_state: int = 42,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance for a fitted model pipeline.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe containing features and target.
    - model_path : str, required
        Path to trained pipeline.
    - output_dir : str, required
        Directory where results will be saved.
    - n_repeats : int, required
        Number of random shuffles per feature.
    - random_state : int, required
        Random seed used for reproducibility.
    - drop_cols : list[str] | None, optional
        Columns to remove before computing importance.

    Returns
    -------
    - importance_df : pd.DataFrame
        Sorted permutation importance table.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    if drop_cols:
        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = load_model(model_path)

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    importances.to_csv(output_path / "permutation_importance.csv", index=False)

    return importances


if __name__ == "__main__":
    data_path = Path("data/processed/processed.csv")
    df = pd.read_csv(data_path)

    importance_df = compute_permutation_importance(
        df=df,
        model_path="models/best_tuned_model.joblib",
        output_dir="models",
        n_repeats=30,
        random_state=42,
    )
    print(importance_df.head(15))