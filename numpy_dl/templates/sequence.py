from typing import List
import numpy as np
from .module import Module


class Sequencer(Module):
    def __init__(self):
        super().__init__()
        self.seq = None

    def add(self, inseq):
        self.seq = inseq

    def forward(self, x: np.array) -> np.array:

        if np.array(x).ndim == 1:
            x = np.expand_dims(x, 0)

        for func in self.seq:
            x = func(x)
        return x

    def backward(self, grad: np.array):
        x = grad
        for func in self.seq[::-1]:
            x = func.backward(x)

    def param(self):
        params = []
        for func in self.seq:
            for param in func.param():
                params.append(param)
        return params

    def param_func(self) -> List[List[np.array]]:
        params = []
        for func in self.seq:
            if len(func.param()) > 0:
                params.append(func)
        return params

    def zero_grad(self):
        if self.seq:
            for func in self.seq:
                func.zero_grad()

    def eval(self):
        if self.seq:
            for func in self.seq:
                func.is_training = False

    def train(self):
        if self.seq:
            for func in self.seq:
                func.is_training = True
