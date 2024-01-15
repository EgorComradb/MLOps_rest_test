import pickle

from sklearn.datasets import load_iris
from sklearn.svm import SVC


def train() -> None:
    iris = load_iris()

    clf = SVC()
    clf.fit(iris.data, iris.target)

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    train()