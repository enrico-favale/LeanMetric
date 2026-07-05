from typing import Optional, Iterable

from pathlib import Path

import numpy as np
import pandas as pd


def add_sex_feature(
    df: pd.DataFrame,
    sex_col: str = "sex",
) -> pd.DataFrame:
    """
    Create a binary sex feature from the gender column. The dataset is composed of only male subjects.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with binary sex feature
    """

    df = df.copy()
    
    df[sex_col] = "Male"
    
    return df


def add_sex_numeric_feature(
    df: pd.DataFrame,
    sex_col: str = "sex",
    sex_numeric_col: str = "sex_numeric",
) -> pd.DataFrame:
    """
    Create a numeric sex feature from the gender column.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - sex_col : str, optional
        Sex column name
    - sex_numeric_col : str, optional
        Numeric sex column name

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with numeric sex feature
    """

    df = df.copy()

    df[sex_numeric_col] = df[sex_col].map({"Male": 1, "Female": 0})

    return df


def add_bmi_feature(
    df: pd.DataFrame,
    weight_col: str = "weight",
    height_col: str = "height",
) -> pd.DataFrame:
    """
    Create BMI feature from weight and height.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - weight_col : str, optional
        Weight column name
    - height_col : str, optional
        Height column name

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with BMI feature
    """

    df = df.copy()

    if weight_col in df.columns and height_col in df.columns:
        df["bmi"] = df[weight_col] / (df[height_col] ** 2) * 10_000
    else:
        raise ValueError(
            f"Columns '{weight_col}' and/or '{height_col}' not found in dataframe."
        )

    return df


def add_relative_circumference_features(
    df: pd.DataFrame,
    height_col: str = "height",
    circumference_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Create circumference-to-height ratio features.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - height_col : str, optional
        Height column name
    - circumference_columns : Optional[Iterable[str]], optional
        Circumference columns to normalize by height

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with new engineered features
    """

    df = df.copy()

    default_columns = [
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

    target_columns = (
        list(circumference_columns)
        if circumference_columns is not None
        else default_columns
    )

    if height_col not in df.columns:
        return df

    for col in target_columns:
        if col in df.columns:
            df[f"{col}_to_height_ratio"] = df[col] / df[height_col]

    return df


def add_body_proportion_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create body proportion features from available circumference columns.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with engineered body proportion features
    """

    df = df.copy()

    if {"chest", "abdomen"}.issubset(df.columns):
        df["abdomen_to_chest_ratio"] = df["abdomen"] / df["chest"]

    if {"thigh", "hip"}.issubset(df.columns):
        df["thigh_to_hip_ratio"] = df["thigh"] / df["hip"]

    if {"biceps", "wrist"}.issubset(df.columns):
        df["biceps_to_wrist_ratio"] = df["biceps"] / df["wrist"]

    return df


def add_age_bins(
    df: pd.DataFrame, 
    age_col: str = "age"
) -> pd.DataFrame:
    """
    Create age group feature from age column.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - age_col : str, optional
        Age column name

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with age group feature
    """

    df = df.copy()

    if age_col in df.columns:
        df["age_group"] = pd.cut(
            df[age_col],
            bins=[0, 29, 39, 49, 59, 120],
            labels=["<30", "30-39", "40-49", "50-59", "60+"],
            include_lowest=True,
        )

    return df


def add_bodyfat_from_bmi_features(
    df: pd.DataFrame, 
    bmi_col: str = "bmi", 
    age_col: str = "age", 
    sex_numeric_col: str = "sex_numeric"
) -> pd.DataFrame:
    """
    Create body fat features from BMI.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with Deuremberg features
    """

    df = df.copy()

    if {bmi_col, age_col, sex_numeric_col}.issubset(df.columns):
        df["deuremberg_bodyfat"] = 1.20 * df[bmi_col] + 0.23 * df[age_col] - 10.8 * df[sex_numeric_col] - 5.4
        df["deuremberg_mod_bodyfat"] = 1.29 * df[bmi_col] + 0.20 * df[age_col] - 11.4 * df[sex_numeric_col] - 8.0
        df["gallagher_bodyfat"] = 1.46 * df[bmi_col] + 0.14 * df[age_col] - 11.6 * df[sex_numeric_col] - 10.0
        df["jackson_pollock_bodyfat"] = 1.61 * df[bmi_col] + 0.13 * df[age_col] - 12.1 * df[sex_numeric_col] - 13.9

    return df


def add_navy_bodyfat_features(
    df: pd.DataFrame,
    abdomen_col: str = "abdomen",
    neck_col: str = "neck",
    hip_col: str = "hip",
    height_col: str = "height",
    sex_numeric_col: str = "sex_numeric"
) -> pd.DataFrame:
    """
    Create navy body fat features from circumference measurements.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - abdomen_col : str, optional
        Abdomen circumference column name
    - neck_col : str, optional
        Neck circumference column name
    - hip_col : str, optional
        Hip circumference column name
    - height_col : str, optional
        Height column name
    - sex_numeric_col : str, optional
        Numeric sex column name, 1 for male and 0 for female

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with navy body fat feature
    """

    df = df.copy()

    required_cols = {
        abdomen_col,
        neck_col,
        hip_col,
        height_col,
        sex_numeric_col,
    }

    if not required_cols.issubset(df.columns):
        return df

    male_mask = df[sex_numeric_col] == 1
    female_mask = df[sex_numeric_col] == 0

    df["navy_bodyfat"] = np.nan

    male_valid_mask = male_mask & ((df[abdomen_col] - df[neck_col]) > 0) & (df[height_col] > 0)
    female_valid_mask = (
        female_mask
        & ((df[abdomen_col] + df[hip_col] - df[neck_col]) > 0)
        & (df[height_col] > 0)
    )

    df.loc[male_valid_mask, "navy_bodyfat"] = (
        495
        / (
            1.0324
            - 0.19077 * np.log10(df.loc[male_valid_mask, abdomen_col] - df.loc[male_valid_mask, neck_col])
            + 0.15456 * np.log10(df.loc[male_valid_mask, height_col])
        )
        - 450
    )

    df.loc[female_valid_mask, "navy_bodyfat"] = (
        495
        / (
            1.29579
            - 0.35004 * np.log10(
                df.loc[female_valid_mask, abdomen_col]
                + df.loc[female_valid_mask, hip_col]
                - df.loc[female_valid_mask, neck_col]
            )
            + 0.22100 * np.log10(df.loc[female_valid_mask, height_col])
        )
        - 450
    )

    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with engineered features
    """

    df = df.copy()

    df = add_sex_numeric_feature(df)
    df = add_bmi_feature(df)
    df = add_relative_circumference_features(df)
    df = add_body_proportion_features(df)
    df = add_age_bins(df)
    df = add_bodyfat_from_bmi_features(df)
    df = add_navy_bodyfat_features(df)
    
    df = df.drop(columns=["sex"])
    
    return df
