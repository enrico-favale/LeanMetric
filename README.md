# LeanMetric: Body Fat Prediction

LeanMetric is a machine learning project for estimating body fat percentage from anthropometric measurements. The pipeline starts from the original BodyFat dataset, applies preprocessing and feature engineering, compares multiple regression models, tunes the best candidates, and exposes a lightweight prediction interface for new raw samples.

LeanMetric is a machine learning project for estimating body fat percentage from anthropometric measurements. The pipeline starts from the Body Fat Prediction Dataset available on Kaggle, applies preprocessing and feature engineering, compares multiple regression models, tunes the best candidates, and exposes a lightweight prediction interface for new raw samples. The dataset reference is: [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset/data).

## Project goals

The goal of the project is to predict `bodyfat` from a small set of anthropometric base measurements such as age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, and wrist. The project is structured like a small ML engineering workflow: data preparation, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, model persistence, and inference.

## Dataset

The starting dataset contains 252 observations and 15 columns. The original variables are `Density`, `BodyFat`, `Age`, `Weight`, `Height`, `Neck`, `Chest`, `Abdomen`, `Hip`, `Thigh`, `Knee`, `Ankle`, `Biceps`, `Forearm`, and `Wrist`. In the project, `Density` is not used as an input feature because it is too close to the target concept and would reduce the realism of the inference task.

The dataset has no missing values and no duplicated rows. The project assumes male subjects in the original dataset and, for inference from raw dictionaries, the `sex` field is handled explicitly as `Male` or `Female` when needed by the derived formulas.

## Repository structure

```text
.
├── data
│   ├── processed
│   │   └── processed.csv
│   └── raw
│       └── bodyfat.csv
├── models
├── notebooks
│   ├── 01_eda_bodyfat.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_train.ipynb
├── src
│   └── leanmetric
│       ├── data
│       │   ├── preprocessing.py
│       │   └── feature_engineering.py
│       └── models
│           ├── evaluate.py
│           ├── inspect.py
│           ├── predict.py
│           ├── registry.py
│           ├── trainer.py
│           └── tune.py
```

## Preprocessing

The preprocessing pipeline standardizes the input dataframe before any modeling step. The main operations are:

- column-name normalization to lower case with underscores;
- duplicate-row removal;
- numeric conversion for selected columns;
- unit conversion from inches to centimeters for `height`;
- unit conversion from pounds to kilograms for `weight`.

These steps are implemented in `src/leanmetric/data/preprocessing.py`. The functions are written to be reusable both in notebook experimentation and in production-style inference.

## Feature engineering

Feature engineering is implemented in `src/leanmetric/data/feature_engineering.py`. Starting from the cleaned base columns, the pipeline creates several derived variables:

- `sex_numeric`, a binary encoding of `sex`;
- `bmi`, computed from weight and height;
- circumference-to-height ratios for the main body measurements;
- proportion features such as `abdomen_to_chest_ratio`, `thigh_to_hip_ratio`, and `biceps_to_wrist_ratio`;
- `age_group`, a discretized age bin feature;
- formula-based body-fat estimates from BMI, including Deuremberg, modified Deuremberg, Gallagher, and Jackson-Pollock variants;
- `navy_bodyfat`, computed using the U.S. Navy formula when the necessary measurements are available.

The full feature-engineering pipeline is executed by `run_feature_engineering()`. During model training, the derived formula-based body-fat columns are treated as candidate features only when explicitly selected, and in the tuned experiments several of them are dropped to avoid leakage-like shortcuts.

## Training workflow

The training process is split into two stages: a broad model comparison stage and a focused tuning stage. The first stage lives in `src/leanmetric/models/trainer.py`, where several regression models are compared on the processed dataset using a train/test split and 5-fold cross-validation. The second stage lives in `src/leanmetric/models/tune.py`, where only the strongest tree-based candidates are tuned with randomized search.

### Baseline model comparison

The comparison stage evaluates these models:

- Linear Regression;
- Ridge Regression;
- Lasso Regression;
- Elastic Net;
- Random Forest Regressor;
- HistGradientBoosting Regressor;
- XGBoost Regressor.

The comparison is done on the processed feature matrix using `MAE`, `RMSE`, and `R²` as regression metrics. The best model from this stage is saved to `models/best_model/` together with its metadata and the summary table of results.

### Hyperparameter tuning

The tuning stage focuses on two final candidates:

- `xgboost`;
- `random_forest`.

`RandomizedSearchCV` is used with `neg_mean_absolute_error` as the optimization objective. The tuned parameter grids are:

#### XGBoost

- `n_estimators`: 200, 300, 400, 600, 800;
- `max_depth`: 2, 3, 4, 5, 6;
- `learning_rate`: 0.01, 0.03, 0.05, 0.1;
- `subsample`: 0.7, 0.8, 0.9, 1.0;
- `colsample_bytree`: 0.7, 0.8, 0.9, 1.0;
- `min_child_weight`: 1, 3, 5, 7;
- `reg_alpha`: 0.0, 0.1, 1.0;
- `reg_lambda`: 0.5, 1.0, 2.0, 5.0.

#### Random Forest

- `n_estimators`: 200, 300, 500, 800;
- `max_depth`: None, 4, 6, 8, 12;
- `min_samples_split`: 2, 5, 10;
- `min_samples_leaf`: 1, 2, 4, 6;
- `max_features`: `sqrt`, 0.5, 0.7, 1.0.

In the tuned top-features experiment, the following columns are dropped before training: `deuremberg_bodyfat`, `deuremberg_mod_bodyfat`, `gallagher_bodyfat`, `jackson_pollock_bodyfat`, and `age_group`.

## Final tuned results

The latest tuning run on the top-features configuration produced the following comparison:

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| XGBoost | 3.7954 | 3.1037 | 3.6378 | 0.7155 |
| Random Forest | 3.9050 | 2.9860 | 3.7241 | 0.7019 |

XGBoost is the best model according to cross-validation MAE, while Random Forest is slightly better on test MAE. Both models remain strong and very close in performance.

## Final models for prediction

The project keeps two final models for inference: `best_xgboost.joblib` and `best_random_forest.joblib`. Both are stored under `models/best_tuned_model_top_features/` and are designed to be used on raw input dictionaries after preprocessing and feature engineering.

The prediction module loads one of these models, normalizes the input dictionary, applies preprocessing, adds the engineered features, drops the columns declared in the corresponding metadata file, and then runs inference. The result includes the predicted body fat from both tuned models together with auxiliary calculations such as BMI and the Navy body-fat estimate.

## Inference

Inference is implemented in `src/leanmetric/models/predict.py`. The intended usage is to pass a single raw sample as a dictionary containing the base anthropometric measurements. The module handles the conversion and feature creation internally, so the caller does not need to build engineered features manually.

Example input:

```python
sample = {
    "age": 23,
    "weight": 77.0,
    "height": 175.0,
    "neck": 38.0,
    "chest": 105.0,
    "abdomen": 91.0,
    "hip": 90.0,
    "thigh": 55.0,
    "knee": 39.0,
    "ankle": 24.0,
    "biceps": 31.0,
    "forearm": 25.0,
    "wrist": 18.0,
    "sex": "Male",
}
```

The prediction function returns a dictionary with the tuned XGBoost prediction, the tuned Random Forest prediction, BMI, and the Navy body-fat estimate.

### Predict from a raw dictionary

```python
from leanmetric.models.predict import predict_bodyfat

pred = predict_bodyfat(sample)
print(pred)
```

## Saved artifacts

The main output folders contain:

- `best_model/`: baseline comparison winner and comparison table;
- `best_model_drop/`: alternative baseline experiment with dropped columns;
- `best_tuned_model/`: tuning results on the full candidate set;
- `best_tuned_model_top_features/`: tuning results on the selected top-features configuration.

The tuned folders include serialized models, metadata JSON files, and the CSV files with the tuning summary.

## Notes

The project is intentionally small and practical. Because the dataset is limited in size, the comparison focuses on regression models that are robust on tabular data rather than on heavy deep-learning architectures. A neural network was considered during experimentation, but the final production-oriented choice remains the pair of tuned tree-based models because they are easier to tune, easier to interpret, and stronger on this dataset.
