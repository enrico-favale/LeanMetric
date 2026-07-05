from __future__ import annotations

import pandas as pd

from leanmetric.models.trainer import train_and_compare


def run_training(data_path: str = "data/processed/processed.csv"):
    """
    Run the full training pipeline.

    Parameters
    ----------
    - data_path : str, required
        Path to the processed dataset.

    Returns
    -------
    - results_df : pd.DataFrame
        Dataframe with model comparison results.
    """
    df = pd.read_csv(data_path)
    results_df, _ = train_and_compare(df=df, output_dir="models", random_state=42)
    return results_df


if __name__ == "__main__":
    results = run_training()
    print(results)