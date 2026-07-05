from typing import Optional, Iterable

from pathlib import Path

import numpy as np
import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with standardized column names
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows from dataframe.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Dataframe without duplicated rows
    """

    df = df.drop_duplicates().copy()

    return df


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Convert selected columns to numeric dtype.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - columns : Optional[Iterable[str]], optional
        Columns to convert. If None, tries to convert None

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with converted numeric columns
    """

    df = df.copy()

    target_columns = columns if columns is not None else []

    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def convert_inches_to_cm(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Convert selected columns from inches to centimeters.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - columns : Optional[Iterable[str]], optional
        Columns to convert. If None, tries to convert all columns

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with converted columns
    """

    df = df.copy()

    target_columns = columns if columns is not None else df.columns

    for col in target_columns:
        if col in df.columns:
            df[col] = df[col] * 2.54

    return df

def convert_lbs_to_kg(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Convert selected columns from pounds to kilograms.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe
    - columns : Optional[Iterable[str]], optional
        Columns to convert. If None, tries to convert all columns

    Returns
    -------
    - df : pd.DataFrame
        Dataframe with converted columns
    """

    df = df.copy()

    target_columns = columns if columns is not None else df.columns

    for col in target_columns:
        if col in df.columns:
            df[col] = df[col] * 0.453592

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Parameters
    ----------
    - df : pd.DataFrame, required
        Input dataframe

    Returns
    -------
    - df : pd.DataFrame
        Preprocessed dataframe
    """

    df = df.copy()

    df = standardize_column_names(df)
    df = remove_duplicates(df)
    df = convert_numeric_columns(df)
    df = convert_inches_to_cm(df, columns=["height"])
    df = convert_lbs_to_kg(df, columns=["weight"])
    
    return df