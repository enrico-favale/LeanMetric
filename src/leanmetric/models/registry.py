from __future__ import annotations

from typing import Dict

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def get_sklearn_models(random_state: int = 42) -> Dict[str, object]:
    """
    Return candidate sklearn regression models.

    Parameters
    ----------
    - random_state : int, required
        Random seed used for reproducibility.

    Returns
    -------
    - models : Dict[str, object]
        Dictionary containing initialized models.
    """
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=10000),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=5,
            l2_regularization=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        ),
        "mlp_sklearn": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
    }


def get_xgb_model(random_state: int = 42):
    """
    Return an initialized XGBoost regressor.

    Parameters
    ----------
    - random_state : int, required
        Random seed used for reproducibility.

    Returns
    -------
    - model : object
        Initialized XGBoost regressor.
    """

    return XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
    )