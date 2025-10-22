import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    def __init__(self, k=3):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        dists = np.linalg.norm(self.X_train - x, axis=1)
        k_idx = np.argsort(dists)[:self.k]
        k_labels = self.y_train[k_idx]
        return Counter(k_labels).most_common(1)[0][0]
