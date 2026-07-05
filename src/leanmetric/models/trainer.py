from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from leanmetric.models.evaluate import regression_metrics
from leanmetric.models.registry import get_sklearn_models, get_xgb_model


TARGET_COL = "bodyfat"


def train_and_compare(
    df: pd.DataFrame,
    output_dir: str = "models",
    random_state: int = 42,
    drop_cols: list[str] | None = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Train and compare multiple regression models on the bodyfat target.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe containing processed features and target.
    - output_dir : str, required
        Directory where training artifacts will be saved.
    - random_state : int, required
        Random seed used for reproducibility.

    Returns
    -------
    - results_df : pd.DataFrame
        Dataframe with model comparison results.
    - best_artifact : dict
        Dictionary containing best model information.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    if drop_cols:
        X = X.drop(columns=drop_cols)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    models = get_sklearn_models(random_state=random_state)
    try:
        models["xgboost"] = get_xgb_model(random_state=random_state)
    except Exception:
        pass

    results = []
    fitted_artifacts = {}

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("model", model),
            ]
        )

        cv_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv)
        cv_metrics = regression_metrics(y_train, cv_pred)

        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)

        artifact = {
            "model_name": model_name,
            "pipeline": pipeline,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }
        fitted_artifacts[model_name] = artifact

        results.append(
            {
                "model": model_name,
                "cv_mae": cv_metrics["mae"],
                "cv_rmse": cv_metrics["rmse"],
                "cv_r2": cv_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )

    results_df = pd.DataFrame(results).sort_values("cv_mae", ascending=True).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_artifact = fitted_artifacts[best_model_name]

    joblib.dump(best_artifact["pipeline"], output_path / "best_model.joblib")
    results_df.to_csv(output_path / "training_results.csv", index=False)

    metadata = {
        "target_column": TARGET_COL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model_name": best_model_name,
        "cv_metrics": best_artifact["cv_metrics"],
        "test_metrics": best_artifact["test_metrics"],
    }

    with open(output_path / "best_model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return results_df, best_artifact