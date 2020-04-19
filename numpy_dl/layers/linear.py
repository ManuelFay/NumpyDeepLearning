from typing import List
import numpy as np
from numpy_dl.templates import Module


class Linear(Module):

    def __init__(self, size_input, size_output):
        super().__init__()
        self.size_input = size_input
        self.size_output = size_output
        # eps = 1e-6
        # eps = 0.1
        eps = 1 / (size_input ** .5)
        self.max_batch_size = 100
        self.weights = np.random.normal(0, eps, (size_input, size_output))
        self.bias = np.random.normal(0, eps, (1, size_output))
        self.input = None
        self.w_grad = np.zeros((self.size_input, self.size_output))
        self.b_grad = np.zeros(self.size_output)

    def forward(self, x: np.array) -> np.array:
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        if self.is_training:
            self.input = x
        else:
            self.input = None
        return np.matmul(x, self.weights) + self.bias

    def backward(self, grad: np.array) -> np.array:
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        grad_input = np.matmul(grad, self.weights.transpose())

        self.b_grad += grad.mean(axis=0)
        self.w_grad += np.einsum('...i,...j', self.input, grad).mean(axis=0)
        return grad_input

    def zero_grad(self):
        self.w_grad = np.zeros(self.w_grad.shape)
        self.b_grad = np.zeros(self.b_grad.shape)

    def sub_grad(self, eta):
        self.weights = self.weights - eta * self.w_grad
        self.bias = self.bias - eta * self.b_grad

    def param(self) -> List[List[np.array]]:
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [[self.weights, self.w_grad], [self.bias, self.b_grad]]
