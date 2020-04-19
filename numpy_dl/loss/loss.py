from typing import List
import numpy as np
from numpy_dl.templates import Module
from numpy_dl.templates import sigmoid


class MSELoss(Module):

    def __init__(self):
        self.target = None
        self.value = None

    def __call__(self, value: np.array, target: np.array) -> float:
        return self.forward(value, target)

    def forward(self, value: np.array, target: np.array, reduction: str = 'sum') -> float:
        # loss function
        self.target = target
        self.value = value

        mse = (target - value) ** 2

        if reduction == 'mean':
            return mse.mean()
        else:
            return mse.sum()

    def backward(self) -> np.array:
        return 2 * (self.value - self.target)


class BCEwithSoftmaxLoss(Module):

    def __init__(self):
        self.target = None
        self.value = None

    def __call__(self, value: np.array, target: np.array) -> float:
        return self.forward(value, target)

    def forward(self, value: np.array, target: np.array, reduction: str ='sum') -> np.array:
        # loss function

        self.target = target
        self.value = value
        bce = -(target * np.log(sigmoid(self.value)) + (1 - target) * np.log(1 - sigmoid(self.value)))

        if reduction == 'mean':
            return bce.mean()
        else:
            return bce.sum()

    def backward(self) -> np.array:
        grad = np.array(self.value - self.target)
        if grad.ndim == 1:
            return np.expand_dims(grad, 1)
        return grad
