import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def _standardize(X):
    sc = StandardScaler()
    return sc.fit_transform(X)

def get_iris_data():
    iris = fetch_ucirepo(id=53)
    X_df = iris.data.features
    y_df = iris.data.targets
    target_col = y_df.columns[0]
    y = y_df[target_col].astype(str).str.replace('Iris-', '', regex=False).to_numpy()
    X = _standardize(X_df.to_numpy(dtype=float))
    return X, y

def get_wine_data():
    wine = fetch_ucirepo(id=109)
    X_df = wine.data.features
    y_df = wine.data.targets
    target_col = 'class' if 'class' in y_df.columns else y_df.columns[0]
    y = y_df[target_col].to_numpy()
    X = _standardize(X_df.to_numpy(dtype=float))
    return X, y
