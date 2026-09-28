import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add sine and cosine components for periodic numeric features."""

    def __init__(self, cyclical_cols=None):
        self.cyclical_cols = cyclical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = X.copy()
        cyclical_cols = self.cyclical_cols or {'hr': 24, 'mnth': 12}

        for col, max_val in cyclical_cols.items():
            if col in transformed.columns:
                transformed[f'{col}_sin'] = np.sin(2 * np.pi * transformed[col] / max_val)
                transformed[f'{col}_cos'] = np.cos(2 * np.pi * transformed[col] / max_val)
        return transformed


def build_model_pipeline(random_state=42, **model_params) -> Pipeline:
    """Construct a leakage-safe preprocessing and regression pipeline.

    Numeric features are standardized for a consistent feature contract and
    future compatibility with scale-sensitive estimators; the current
    histogram gradient boosting regressor itself is scale-invariant.
    """
    categorical_cols = ['season', 'weathersit', 'weekday']
    numeric_cols = [
        'yr', 'mnth', 'hr', 'holiday', 'workingday',
        'temp', 'hum', 'windspeed',
        'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('cyclical', CyclicalFeatureTransformer(cyclical_cols={'hr': 24, 'mnth': 12})),
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=random_state, **model_params))
    ])

    return pipeline