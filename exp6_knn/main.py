import numpy as np
import matplotlib.pyplot as plt
from data import get_iris_data, get_wine_data
from utils import train_test_split
from knn_classifier import KNNClassifier

def evaluate_dataset(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ks = [1, 3, 5, 7, 9, 11, 15]
    accs = []
    for k in ks:
        clf = KNNClassifier(k=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accs.append(acc)
        print(f"K={k:<2} -> Accuracy: {acc*100:.2f}%")
    best_idx = int(np.argmax(accs))
    best_k = ks[best_idx]
    best_acc = accs[best_idx]
    print(f"Best K for {dataset_name}: {best_k} (Accuracy: {best_acc*100:.2f}%)")
    plt.figure()
    plt.plot(ks, accs, marker="o")
    plt.title(f"Accuracy vs K ({dataset_name})")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
    return best_k, best_acc

if __name__ == "__main__":
    X_iris, y_iris = get_iris_data()
    best_k_iris, best_acc_iris = evaluate_dataset(X_iris, y_iris, "Iris")
    X_wine, y_wine = get_wine_data()
    best_k_wine, best_acc_wine = evaluate_dataset(X_wine, y_wine, "Wine")
    print("\nSummary")
    print(f"Iris -> Best k={best_k_iris}, Accuracy={best_acc_iris*100:.2f}%")
    print(f"Wine -> Best k={best_k_wine}, Accuracy={best_acc_wine*100:.2f}%")
