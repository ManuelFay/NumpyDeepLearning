import numpy as np
from numpy_dl.templates import Module


class Linear(Module):

    def __init__(self, size_input, size_output):
        self.size_input = size_input
        self.size_output = size_output
        # eps = 1e-6
        # eps = 0.1
        eps = 1 / (size_input ** .5)
        self.weights = np.random.normal(0, eps, (size_input, size_output))
        self.bias = np.random.normal(0, eps, (size_output, 1))
        self.input = None
        self.w_grad = np.zeros((size_input, size_output))
        self.b_grad = np.zeros((size_output, 1))

    def forward(self, x):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        self.input = x
        return np.matmul(x.reshape(1, -1), self.weights).reshape(-1, 1) + self.bias

    def backward(self, grad):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        grad_input = np.matmul(self.weights, grad)

        self.b_grad += grad
        self.w_grad += np.matmul(grad, self.input.reshape(1, -1)).transpose()
        return grad_input

    def zero_grad(self):
        self.w_grad = np.zeros((self.size_input, self.size_output))
        self.b_grad = np.zeros((self.size_output, 1))

    def sub_grad(self, eta):
        self.weights = self.weights - eta * self.w_grad
        self.bias = self.bias - eta * self.b_grad

    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [[self.weights, self.w_grad], [self.bias, self.b_grad]]
