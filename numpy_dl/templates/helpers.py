import numpy as np


def sigmoid(x: np.array, derivative=False) -> np.array:
    ret = x*(1-x) if derivative else np.power((1+np.exp(-x)), -1)
    return ret


def stable_softmax(X: np.array) -> np.array:
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def evaluate_accuracy(x: np.array, y: np.array) -> float:
    if x.shape[1] == 1:
        return np.mean((y > 0) == (x > 0)).item()

    return np.mean(y.squeeze().argmax(dim=1) == x.argmax(dim=1)).item()
