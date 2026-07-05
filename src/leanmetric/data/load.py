from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os

import pandas as pd

def get_path_from_env(variable_name : str) -> Path:
    """
    Returns the path of a variable from the .env file.

    Parameters
    ----------
    - variable_name : str, required
        The name of the environment variable containing the path

    Returns
    -------
    - path : Path
        The path of the variable
    """

    load_dotenv(find_dotenv(usecwd=True))
    
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("PROJECT_ROOT non definito nel .env")

    variable_path = os.getenv(variable_name)
    if not variable_path:
        raise ValueError(f"{variable_name} non definito nel .env")

    VARIABLE_PATH = Path(project_root) / Path(variable_path)

    if not VARIABLE_PATH.exists():
        raise FileNotFoundError(f"File non trovato: {VARIABLE_PATH}")
    
    return VARIABLE_PATH


def load_data(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    - path : Path, required
        Path to the CSV file.

    Returns
    -------
    - df : pd.DataFrame
        Loaded dataset.
    """
    
    df = pd.read_csv(path)
    
    return df