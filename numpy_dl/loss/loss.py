import numpy as np
from numpy_dl.templates import Module
from numpy_dl.templates import sigmoid


class MSELoss(Module):

    def __init__(self):
        self.t = None
        self.v = None

    def __call__(self, v, t):
        return self.forward(v, t)

    def forward(self, v, t):
        # loss function
        self.t = t
        self.v = v
        # print(t.shape,v.shape)
        return ((t - v) ** 2).sum()

    def backward(self):
        # dloss
        # d((t1-v1)^2 + (t2-v2)^2 +(t3-v3)^2 + ...)/dvi = d/dvi (ti^2 - 2tivi + vi^2) = 2(vi-ti)

        return 2 * (self.v - self.t)


class BCELoss(Module):

    def __init__(self):
        self.t = None
        self.v = None

    def __call__(self, v, t):
        return self.forward(v, t)

    def forward(self, v, t):
        # loss function
        self.v = v
        self.t = t

        return (-(t * np.log(sigmoid(v)) + (1 - t) * np.log(1 - sigmoid(v))))

    def backward(self):
        # dloss
        # d((t1-v1)^2 + (t2-v2)^2 +(t3-v3)^2 + ...)/dvi = d/dvi (ti^2 - 2tivi + vi^2) = 2(vi-ti)

        return self.t * (sigmoid(self.v) - 1) + (1 - self.t) * sigmoid(self.v)
