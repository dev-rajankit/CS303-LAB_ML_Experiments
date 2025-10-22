# data.py
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    """Load and preprocess the Iris dataset."""
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features)
    y = pd.DataFrame(iris.data.targets)

    y.columns = ['species']
    y['species'] = y['species'].str.replace('Iris-', '', regex=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values.ravel()


def load_wine_data():
    """Load and preprocess the Wine dataset."""
    wine = fetch_ucirepo(id=109)
    X = pd.DataFrame(wine.data.features)
    y = pd.DataFrame(wine.data.targets)
    y.columns = ['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values.ravel()
