from pathlib import Path

import joblib
import pandas as pd


FEATURE_DROP_COLS = ['instant', 'dteday', 'casual', 'registered', 'cnt', 'atemp']
REQUIRED_FEATURE_COLS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'hum', 'windspeed'
]


def load_saved_model(model_path: str | Path = 'best_model.joblib'):
    """Load the fitted model from a persisted model artifact."""
    artifact = joblib.load(model_path)
    if 'model' not in artifact:
        raise ValueError(f'Model artifact does not contain a fitted model: {model_path}')
    return artifact['model']


def predict_and_save(
    dataset: pd.DataFrame,
    model_path: str | Path = 'best_model.joblib',
    output_path: str | Path = 'data/predictions.csv',
    prediction_column: str = 'cnt',
) -> pd.DataFrame:
    """Predict demand for feature-only data and save the result as CSV."""
    model = load_saved_model(model_path)
    missing_columns = [
        column for column in REQUIRED_FEATURE_COLS if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(f'Dataset is missing required feature columns: {missing_columns}')

    features = dataset.drop(
        columns=[column for column in FEATURE_DROP_COLS if column in dataset.columns]
    )
    predictions = model.predict(features)
    result = dataset.copy()
    result[prediction_column] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
