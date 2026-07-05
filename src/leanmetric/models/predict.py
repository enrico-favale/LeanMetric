from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import json

from leanmetric.data.feature_engineering import run_feature_engineering
from leanmetric.data.preprocessing import standardize_column_names, remove_duplicates
from leanmetric.data.load import get_path_from_env

MODEL_DIR_PATH = get_path_from_env("MODEL_DIR_PATH")
BEST_TUNED_MODEL_TOP_FEATURES_PATH = MODEL_DIR_PATH / "best_tuned_model_top_features"

XGBOOST_MODEL_PATH = BEST_TUNED_MODEL_TOP_FEATURES_PATH / "best_xgboost.joblib"
RANDOM_FOREST_MODEL_PATH = BEST_TUNED_MODEL_TOP_FEATURES_PATH / "best_random_forest.joblib"

DEFAULT_MODEL_PATHS = {
    "xgboost": XGBOOST_MODEL_PATH,
    "random_forest": RANDOM_FOREST_MODEL_PATH,
}

BASE_FEATURES = [
    "age",
    "weight",
    "height",
    "neck",
    "chest",
    "abdomen",
    "hip",
    "thigh",
    "knee",
    "ankle",
    "biceps",
    "forearm",
    "wrist",
]


def load_model(model_name: str, model_path: str | Path | None = None) -> Any:
    """
    Load a trained model pipeline.

    Parameters
    ----------
    - model_name : str, required
        Model alias, either 'random_forest' or 'xgboost'.
    - model_path : str | Path | None, optional
        Custom path to the serialized model.

    Returns
    -------
    - model : Any
        Loaded model pipeline.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATHS[model_name]

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def normalize_input_dict(sample: dict[str, Any]) -> pd.DataFrame:
    """
    Build a one-row dataframe from a raw input dictionary.

    Parameters
    ----------
    - sample : dict[str, Any], required
        Raw anthropometric sample.

    Returns
    -------
    - df : pd.DataFrame
        One-row dataframe with standardized columns.
    """
    df = pd.DataFrame([sample]).copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def prepare_features(sample: dict[str, Any]) -> pd.DataFrame:
    """
    Apply preprocessing and feature engineering starting from base features.

    Parameters
    ----------
    - sample : dict[str, Any], required
        Raw anthropometric sample.

    Returns
    -------
    - df : pd.DataFrame
        Engineered feature dataframe ready for prediction.
    """
    df = normalize_input_dict(sample)

    if "density" in df.columns:
        df = df.drop(columns=["density"])

    if "bodyfat" in df.columns:
        df = df.drop(columns=["bodyfat"])

    if "sex" not in df.columns:
        df["sex"] = "Male"

    df = run_feature_engineering(df)

    if "sex" in df.columns:
        df = df.drop(columns=["sex"])

    return df


def drop_unnecessary_columns(
    df: pd.DataFrame, 
    model_name: str = "random_forest"
) -> pd.DataFrame:
    """
    Drop unnecessary columns from the input dataframe.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe.
    - model_name : str, required
        Model alias, either 'random_forest' or 'xgboost'.

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with unnecessary columns dropped.
    """
    
    df = df.copy()

    with open(BEST_TUNED_MODEL_TOP_FEATURES_PATH / f"best_{model_name}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dropped_columns = metadata["dropped_columns"]
    
    return df.drop(columns=dropped_columns, errors="ignore")


def calculate_bmi(df: pd.DataFrame, weight_col: str = "weight", height_col: str = "height") -> float:
    """
    Calculate BMI from weight and height columns.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe.
    - weight_col : str, optional
        Weight column name.
    - height_col : str, optional
        Height column name.

    Returns
    -------
    - bmi : float
        Calculated BMI value.
    """

    bmi = df[weight_col].iloc[0] / (df[height_col].iloc[0] / 100) ** 2
    
    return float(bmi)

def calculate_navy_bodyfat(df: pd.DataFrame) -> float:
    """
    Calculate body fat percentage using the U.S. Navy method.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe with necessary measurements.

    Returns
    -------
    - bodyfat : float
        Calculated body fat percentage.
    """
    
    if df["sex_numeric"].iloc[0] == 1:
        bodyfat = 495 / (1.0324 - 0.19077 * np.log10(df["abdomen"].iloc[0] - df["neck"].iloc[0]) + 0.15456 * np.log10(df["height"].iloc[0])) - 450
    else:
        bodyfat = 495 / (1.29579 - 0.35004 * np.log10(df["abdomen"].iloc[0] + df["hip"].iloc[0] - df["neck"].iloc[0]) + 0.221 * np.log10(df["height"].iloc[0])) - 450

    return float(bodyfat)


def predict_bodyfat(
    sample: dict[str, Any],
) -> dict[str, float]:
    """
    Predict body fat percentage from a raw input dictionary.

    Parameters
    ----------
    - sample : dict[str, Any], required
        Raw anthropometric sample.
    - model_name : str, required
        Model alias, either 'random_forest' or 'xgboost'.
    - model_path : str | Path | None, optional
        Custom path to the serialized model.

    Returns
    -------
    - prediction : dict[str, float]
        Predicted body fat percentage for each model.
    """
    model_xg = load_model(model_name="xgboost", model_path=XGBOOST_MODEL_PATH)
    
    X = prepare_features(sample)
    X = drop_unnecessary_columns(X, model_name="xgboost")
    
    pred_xg = model_xg.predict(X)
    
    model_rf = load_model(model_name="random_forest", model_path=RANDOM_FOREST_MODEL_PATH)
    
    X = prepare_features(sample)
    X = drop_unnecessary_columns(X, model_name="random_forest")
    
    pred_rf = model_rf.predict(X)
    
    bmi = calculate_bmi(X)
    navy_bodyfat = calculate_navy_bodyfat(X)
    
    res_dict = {
        "predicted_bodyfat_xgboost": float(np.asarray(pred_xg).ravel()[0]),
        "predicted_bodyfat_random_forest": float(np.asarray(pred_rf).ravel()[0]),
        "bmi": bmi,
        "navy_bodyfat": navy_bodyfat,
    }
    
    return res_dict