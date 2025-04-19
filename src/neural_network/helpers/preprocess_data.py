import json
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline

from .config import PIPELINE, SCALER

def save_pipeline(p: Pipeline) -> None:
    '''
    Save pipeline configs to file in JSON format.
    '''

    minmax: MinMaxScaler = p.named_steps['scaler']
    polynomial_features: PolynomialFeatures = p.named_steps['polynomial']
    with open(PIPELINE, 'w') as file:
        json.dump({
                'scaler':
                    {
                        'min': minmax.min_.tolist(),
                        'scale': minmax.scale_.tolist(),
                        'feature_range': minmax.feature_range,
                    },
                'polynomial':
                    {
                        'degree': polynomial_features.degree,
                        'include_bias': polynomial_features.include_bias,
                    },
            },
            fp=file,
            indent=4
        )

def save_target_scaler(Y_scaler: MinMaxScaler) -> None:
    with open(SCALER, 'w') as file:
        json.dump({
                'scaler':
                    {
                        'min': Y_scaler.min_.tolist(),
                        'scale': Y_scaler.scale_.tolist(),
                        'feature_range': Y_scaler.feature_range,
                    },
            },
            fp=file,
            indent=4
        )

def transform_sin(X):
    '''
    Add sin features.
    '''

    X_sin = np.sin(X)
    return np.concatenate([X, X_sin], axis=1)

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Split data into X and Y
    X_train, Y_train = data.drop(['target'], axis=1), data['target']
    # Create pipeline for features processing, fit and transform features
    p = Pipeline([
        ('scaler', MinMaxScaler()),
        ('polynomial', PolynomialFeatures(2, include_bias=False)),
        ('functional_transformer', FunctionTransformer(transform_sin))
    ])
    p.fit(X_train)
    X_train = p.transform(X_train)
    # Preprocess target variable
    Y_scaler = MinMaxScaler()
    Y_scaler.fit(Y_train.to_numpy().reshape(-1, 1))
    Y_train = Y_scaler.transform(Y_train.to_numpy().reshape(-1, 1))
    # Save pipeline configs and target variable scaler configs
    save_pipeline(p)
    save_target_scaler(Y_scaler)
    return X_train, Y_train