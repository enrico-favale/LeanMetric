from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from leanmetric.models.registry import get_xgb_model


TARGET_COL = "bodyfat"

DEFAULT_DROP_COLS = [
    "deuremberg_bodyfat",
    "deuremberg_mod_bodyfat",
    "gallagher_bodyfat",
    "jackson_pollock_bodyfat",
    "age_group",
]


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
    }


def make_param_distributions(random_state: int = 42) -> Dict[str, Dict[str, list]]:
    """
    Create parameter distributions for tuning.

    Parameters
    ----------
    - random_state : int, required
        Random seed used for reproducibility.

    Returns
    -------
    - param_spaces : Dict[str, Dict[str, list]]
        Parameter search spaces per model.
    """
    xgb_space = {
        "model__n_estimators": [200, 300, 400, 600, 800],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    rf_space = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 4, 6, 8, 12],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
    }

    return {
        "xgboost": xgb_space,
        "random_forest": rf_space,
    }


def save_model_artifacts(
    model_name: str,
    search: RandomizedSearchCV,
    test_metrics: Dict[str, float],
    output_path: Path,
    target_column: str,
    dropped_columns: list[str],
) -> None:
    """
    Save tuned model and its metadata.

    Parameters
    ----------
    - model_name : str, required
        Model identifier.
    - search : RandomizedSearchCV, required
        Fitted randomized search object.
    - test_metrics : Dict[str, float], required
        Test metrics for the best estimator.
    - output_path : Path, required
        Output directory path.
    - target_column : str, required
        Name of target column.
    - dropped_columns : list[str], required
        Columns removed before training.

    Returns
    -------
    - None : type
        Model artifacts are saved to disk.
    """
    model_file = output_path / f"best_{model_name}.joblib"
    metadata_file = output_path / f"best_{model_name}_metadata.json"

    joblib.dump(search.best_estimator_, model_file, compress=3)

    metadata = {
        "model_name": model_name,
        "target_column": target_column,
        "dropped_columns": dropped_columns,
        "best_cv_mae": float(-search.best_score_),
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)


def tune_models(
    df: pd.DataFrame,
    output_dir: str = "models",
    random_state: int = 42,
    drop_cols: list[str] | None = None,
    n_iter: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tune XGBoost and Random Forest models with randomized search.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe with features and target.
    - output_dir : str, required
        Directory where artifacts will be saved.
    - random_state : int, required
        Random seed used for reproducibility.
    - drop_cols : list[str] | None, optional
        Columns to remove before training.
    - n_iter : int, required
        Number of sampled parameter combinations.

    Returns
    -------
    - results_df : pd.DataFrame
        Dataframe with tuning results.
    - best_artifact : Dict[str, Any]
        Dictionary with overall best model information.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    dropped_columns = []
    if drop_cols is not None:
        drop_cols = list(drop_cols)
        dropped_columns = [c for c in drop_cols if c in df.columns and c != TARGET_COL]
        if dropped_columns:
            df = df.drop(columns=dropped_columns)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    param_spaces = make_param_distributions(random_state=random_state)

    candidates = {
        "xgboost": get_xgb_model(random_state=random_state),
        "random_forest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = []
    fitted = {}

    for model_name, model in candidates.items():
        pipe = Pipeline(
            steps=[
                ("model", model),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_spaces[model_name],
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_pred = best_model.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)

        results.append(
            {
                "model": model_name,
                "best_cv_mae": float(-search.best_score_),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "best_params": search.best_params_,
            }
        )

        fitted[model_name] = {
            "search": search,
            "best_estimator": best_model,
            "test_metrics": test_metrics,
        }

        save_model_artifacts(
            model_name=model_name,
            search=search,
            test_metrics=test_metrics,
            output_path=output_path,
            target_column=TARGET_COL,
            dropped_columns=dropped_columns,
        )

    results_df = pd.DataFrame(results).sort_values("best_cv_mae", ascending=True).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_artifact = fitted[best_model_name]

    joblib.dump(best_artifact["best_estimator"], output_path / "best_tuned_model.joblib", compress=3)

    results_to_save = results_df.copy()
    results_to_save["best_params"] = results_to_save["best_params"].apply(lambda x: json.dumps(x, default=str))
    results_to_save.to_csv(output_path / "tuning_results.csv", index=False)

    overall_metadata = {
        "target_column": TARGET_COL,
        "dropped_columns": dropped_columns,
        "best_model_name": best_model_name,
        "best_cv_mae": float(results_df.iloc[0]["best_cv_mae"]),
        "best_test_metrics": best_artifact["test_metrics"],
        "best_params": results_df.iloc[0]["best_params"],
    }

    with open(output_path / "best_tuned_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(overall_metadata, f, indent=4, default=str)

    return results_df, best_artifact


if __name__ == "__main__":
    data_path = Path("data/processed/processed.csv")
    df = pd.read_csv(data_path)
    results_df, _ = tune_models(
        df=df,
        output_dir="models",
        drop_cols=DEFAULT_DROP_COLS,
        random_state=42,
        n_iter=30,
    )
    print(results_df)