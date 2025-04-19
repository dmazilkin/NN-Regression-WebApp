import json
import numpy as np
import pandas as pd
from keras.api.saving import load_model
from keras import Sequential
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, FunctionTransformer

from .helpers.preprocess_data import transform_sin
from .helpers.config import TRAINED_MODEL, PIPELINE, SCALER

class NeuralNetwork:
    def __init__(self) -> None:
        # Load trained neural network
        self._nn: Sequential = load_model(TRAINED_MODEL)
        self._x_scaler: MinMaxScaler = self._load_scaler(PIPELINE)
        self._x_polynomial: PolynomialFeatures = self._load_polynomial(PIPELINE)
        self._x_functional_transform: FunctionTransformer = self._load_functional_transformer()
        self._y_scaler: MinMaxScaler = self._load_scaler(SCALER)

    def predict(self, X):
        df = pd.DataFrame(X)
        # Apply MinMaxScaler
        df = self._x_scaler.transform(df)
        # Apply polynomial features
        self._x_polynomial.fit(df)
        df = self._x_polynomial.transform(df)
        # Apply sin features
        self._x_functional_transform.fit(df)
        df = self._x_functional_transform.transform(df)
        predict = self._nn.predict(df, verbose=0)
        return self._y_scaler.inverse_transform(predict).reshape(1, -1).tolist()

    @staticmethod
    def _load_polynomial(path: str):
        with open(path, 'r') as file:
            configs = json.load(file)
        polynomial_configs = configs['polynomial']
        polynomial = PolynomialFeatures(degree=polynomial_configs['degree'], include_bias=polynomial_configs['include_bias'])
        return polynomial

    @staticmethod
    def _load_functional_transformer():
        return FunctionTransformer(transform_sin)

    @staticmethod
    def _load_scaler(path: str):
        with open(path, 'r') as file:
            configs = json.load(file)
        scaler_configs = configs['scaler']
        scaler = MinMaxScaler(feature_range=tuple(scaler_configs['feature_range']))
        scaler.min_ = scaler_configs['min']
        scaler.scale_ = scaler_configs['scale']
        return scaler

