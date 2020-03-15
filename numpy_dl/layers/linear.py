import numpy as np
from numpy_dl.templates import Module


class Linear(Module):

    def __init__(self, sizeInput, sizeOutput):
        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        # eps = 1e-6
        # eps = 0.1
        eps = 1 / (sizeInput ** .5)
        self.w = np.random.normal(0, eps, (sizeInput, sizeOutput))
        self.b = np.random.normal(0, eps, (sizeOutput, 1))
        self.input = None
        self.wgrad = np.zeros((sizeInput, sizeOutput))
        self.bgrad = np.zeros((sizeOutput, 1))

    def forward(self, x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return np.matmul(x.reshape(1, -1), self.w).reshape(-1, 1) + self.b

    def backward(self, gradwrtoutput):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        gradwrtinput = np.matmul(self.w, gradwrtoutput)

        self.bgrad += gradwrtoutput
        self.wgrad += np.matmul(gradwrtoutput, self.input.reshape(1, -1)).transpose()
        # self.wgrad += np.matmul(gradwrtoutput.reshape(-1, 1),self.input.reshape(1,-1)).t()

        return gradwrtinput

    def zero_grad(self):
        self.wgrad = np.zeros((self.sizeInput, self.sizeOutput))
        self.bgrad = np.zeros((self.sizeOutput, 1))

    def sub_grad(self, eta):
        self.w = self.w - eta * self.wgrad
        self.b = self.b - eta * self.bgrad

    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [[self.w, self.wgrad], [self.b, self.bgrad]]
