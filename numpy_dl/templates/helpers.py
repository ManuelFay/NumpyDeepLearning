import numpy as np


def sigmoid(x, derivative=False):
    ret = x*(1-x) if derivative else np.power((1+np.exp(-x)), -1)
    return ret


def evaluate_accuracy(x, y):
    if x.shape[1] == 1:
        return np.mean((y > 0) == (x > 0))

    return (y.squeeze().argmax(dim=1) == x.argmax(dim=1)).mean()
