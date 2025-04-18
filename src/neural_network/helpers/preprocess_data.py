import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline

def preprocess_data(data: pd.DataFrame):
    X_train, Y_train = data.drop(['target'], axis=1), data['target']

    def add_sin(X):
        X_sin = np.sin(X)
        return np.concatenate([X, X_sin], axis=1)

    p = make_pipeline(MinMaxScaler(), PolynomialFeatures(2, include_bias=False), FunctionTransformer(add_sin))
    p.fit(X_train)
    X_train = p.transform(X_train)
    Y_scaler = MinMaxScaler()
    Y_scaler.fit(Y_train.to_numpy().reshape(-1, 1))
    Y_train = Y_scaler.transform(Y_train.to_numpy().reshape(-1, 1))

    return X_train, Y_train