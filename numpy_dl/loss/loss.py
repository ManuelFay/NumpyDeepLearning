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

    def forward(self, value: np.array, target: np.array) -> float:
        # loss function
        self.target = target
        self.value = value
        return ((target - value) ** 2).sum()

    def backward(self) -> np.array:
        return 2 * (self.value - self.target)


class BCEwithSoftmaxLoss(Module):

    def __init__(self):
        self.target = None
        self.value = None

    def __call__(self, value: np.array, target: np.array) -> float:
        return self.forward(value, target)

    def forward(self, value: np.array, target: np.array) -> float:
        # loss function
        self.target = target
        self.value = value

        return -(target * np.log(sigmoid(self.value)) + (1 - target) * np.log(1 - sigmoid(self.value)))

    def backward(self) -> np.array:
        return self.value - self.target
