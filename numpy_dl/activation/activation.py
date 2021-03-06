from typing import List
import numpy as np
from numpy_dl.templates import Module
from numpy_dl.templates import sigmoid


class Tanh(Module):

    def __init__(self):
        super().__init__()
        # maybe initialize a tensor to keep in memory the inputs?
        self.input = None

    def forward(self, x: np.array) -> np.array:
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        if self.is_training:
            self.input = x
        else:
            self.input = None
        return np.tanh(x)

    def backward(self, grad: np.array) -> np.array:
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        # return (self.input.cosh().power(-2))*x
        return (4 * (np.exp(self.input) + np.exp(self.input * (-1))) ** (-2)) * grad

    def param(self) -> List:
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []


class Sigmoid(Module):

    def __init__(self):
        super().__init__()
        # maybe initialize a tensor to keep in memory the inputs?
        self.input = None

    def forward(self, x: np.array) -> np.array:
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        if self.is_training:
            self.input = x
        else:
            self.input = None
        return sigmoid(x)

    def backward(self, grad: np.array) -> np.array:
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        # return (self.input.cosh().power(-2))*x
        return sigmoid(grad, derivative=True)

    def param(self) -> List:
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []


class ReLU(Module):

    def __init__(self):
        super().__init__()
        # maybe initialize a tensor to keep in memory the inputs?
        self.input = None

    def forward(self, x: np.array) -> np.array:
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        if self.is_training:
            self.input = x
        else:
            self.input = None
        return (x >= 0).astype('float') * x

    def backward(self, grad: np.array) -> np.array:
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        # 1 if >0, 0 otherwise
        return ((self.input >= 0).astype('float')) * grad

    def param(self) -> List:
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return []
