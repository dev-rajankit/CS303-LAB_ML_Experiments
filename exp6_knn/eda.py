import numpy as np
import matplotlib.pyplot as plt
from data import get_iris_data, get_wine_data

def explore_iris():
    X, y = get_iris_data()
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    classes = np.unique(y)
    colors = ['crimson', 'seagreen', 'royalblue']
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for ax, (i, j) in zip(axes.ravel(), pairs):
        for cls, c in zip(classes, colors):
            m = (y == cls)
            ax.scatter(X[m, i], X[m, j], s=20, alpha=0.7, color=c, label=cls)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(f"{names[i]} vs {names[j]}")
        ax.legend()
    fig.suptitle("Iris: pairwise feature plots")
    fig.tight_layout()
    plt.show()

def explore_wine():
    X, y = get_wine_data()
    names = [
        "Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium",
        "Total phenols","Flavanoids","Nonflavanoid phenols",
        "Proanthocyanins","Color intensity","Hue",
        "OD280/OD315 of diluted wines","Proline"
    ]
    classes = np.unique(y)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    pairs = [(0,1), (0,9), (6,9), (9,12)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (i, j) in zip(axes.ravel(), pairs):
        for cls, c in zip(classes, cmap):
            m = (y == cls)
            ax.scatter(X[m, i], X[m, j], s=20, alpha=0.7, color=c, label=f"Class {cls}")
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(f"{names[i]} vs {names[j]}")
        ax.legend()
    fig.suptitle("Wine: selected feature pairs")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_iris()
    explore_wine()
