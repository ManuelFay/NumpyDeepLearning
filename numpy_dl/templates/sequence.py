from .module import Module


class Sequencer(Module):
    def __init__(self):
        self.seq = None

    def add(self, inseq):
        self.seq = inseq

    def forward(self, x):

        for func in self.seq:
            x = func(x)
        return x

    def backward(self, grad):
        x = grad
        for func in self.seq[::-1]:
            x = func.backward(x)

    def param(self):
        params = []
        for func in self.seq:
            for p in func.param():
                params.append(p)
        return params

    def paramFunc(self):
        params = []
        for func in self.seq:
            if len(func.param()) > 0:
                params.append(func)
        return params

    def zero_grad(self):
        if self.seq:
            for func in self.seq:
                func.zero_grad()
